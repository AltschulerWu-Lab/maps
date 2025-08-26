import pandas as pd
import numpy as np
import json
import torch
import sys
from pathlib import Path
from sklearn.metrics import roc_auc_score

# Add maps to path if needed
sys.path.append("/home/kkumbier/maps/")

from maps.screens import ImageScreenMultiAntibody
from maps.multiantibody.config import TrainConfig, ModelConfig, DataLoaderConfig
import maps.multiantibody.data_loaders as data_loaders
import maps.multiantibody.models as models
import maps.multiantibody.training as training


def run_multimarker_binary_analysis(
    train_screen,
    test_screen,
    antibodies=("HSP70/SOD1", "FUS/EEA1"),
    n_classes=2,
    batch_size=6,
    n_epochs=150,
    patience=15,
    lr=1e-3,
    d_model=16,
    n_layers=0,
    random_seed=None
):
    """
    Run the multimarker binary classification analysis.
    
    Parameters:
    -----------
    train_screen : ImageScreenMultiAntibody
        Preprocessed training screen
    test_screen : ImageScreenMultiAntibody
        Preprocessed test screen
    antibodies : tuple
        Antibody markers to use
    n_classes : int
        Number of classes
    batch_size : int
        Batch size for training
    n_epochs : int
        Maximum number of training epochs
    patience : int
        Early stopping patience
    lr : float
        Learning rate
    d_model : int
        Model dimension
    n_layers : int
        Number of layers
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (results_dict, predictions_df) where results_dict contains metrics 
        and predictions_df contains cell line predictions
    """
    
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # --- Setup ---
    # Initialize configurations
    dataloader_config = DataLoaderConfig()
    train_config = TrainConfig()
    model_config = ModelConfig()
    model_config.n_classes = n_classes

    # Create dataloaders
    dataloader_config.batch_size = batch_size
    train_dataloader = data_loaders.create_multiantibody_dataloader(
        train_screen, **vars(dataloader_config)
    )

    scalers = train_dataloader._get_scalers()
    dataloader_config.mode = "eval"
    test_dataloader = data_loaders.create_multiantibody_dataloader(
        test_screen, scalers=scalers, **vars(dataloader_config)
    )

    # Configure and train model
    train_config.n_epochs = n_epochs
    train_config.patience = patience
    train_config.lr = lr

    model_config.antibody_feature_dims = train_dataloader._get_feature_dims()
    model_config.d_model = d_model
    model_config.n_layers = n_layers

    # --- Training ---
    model = models.MultiAntibodyClassifier(**vars(model_config))
    training.train(model, train_dataloader, train_config)

    # --- Generate predictions ---
    model.eval()
    probs_line, probs_cell, labels, cell_lines = [], {}, [], []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in test_dataloader:
            if batch is None:
                continue
            
            x_dict = {ab: batch[ab][0].to(device) for ab in batch}
            y_line = batch[list(batch.keys())[0]][1].to(device)
            cell_lines.extend([batch[list(batch.keys())[0]][-1]])

            cell_logits, line_logits = model(x_dict)        
            
            labels.append(y_line.cpu())
            probs_line.append(torch.softmax(line_logits, dim=-1).cpu())

            for ab in cell_logits:
                if ab not in probs_cell:
                    probs_cell[ab] = []        
                probs_cell[ab].append(torch.softmax(cell_logits[ab], dim=-1).cpu())

    # Merge outputs over cell lines
    probs_line = torch.cat(probs_line)           
    labels = torch.cat(labels)
    probs_cell = {k: torch.cat(v).mean(dim=1) for k, v in probs_cell.items()}

    # Create DataFrame with aggregate class probabilities from probs_line
    df = pd.DataFrame(probs_line.numpy())
    df.columns = [f"class_{i}_agg" for i in df.columns]

    # Add per-antibody class probabilities from probs_cell
    for ab in probs_cell:
        dfab = pd.DataFrame(probs_cell[ab].numpy())
        dfab.columns = [f"class_{i}_{ab}" for i in dfab.columns]
        df = pd.concat([df, dfab], axis=1)

    df["CellLines"] = cell_lines
    df["True"] = labels.numpy()

    # Compute entropy-based weights from per-antibody predictions
    entropy_array = np.zeros((len(cell_lines), len(antibodies)))
    
    for i, ab in enumerate(antibodies):
        probs = probs_cell[ab].numpy()
        epsilon = 1e-8
        entropy = -np.sum(probs * np.log(probs + epsilon), axis=1)
        entropy_array[:, i] = entropy

    # Convert entropies to weights
    max_entropy = np.log(n_classes)
    weights = (max_entropy - entropy_array) / max_entropy
    weights = np.maximum(weights, 0)
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = np.where(row_sums > 0, weights / row_sums, 1.0 / len(antibodies))

    # Compute weighted average predictions for all classes
    weighted_avg_probs = np.zeros((len(cell_lines), n_classes))
    
    for class_idx in range(n_classes):
        class_probs = np.zeros((len(cell_lines), len(antibodies)))
        for i, ab in enumerate(antibodies):
            class_probs[:, i] = probs_cell[ab].numpy()[:, class_idx]
        weighted_avg_probs[:, class_idx] = np.sum(weights * class_probs, axis=1)

    # Add weighted probabilities to DataFrame
    for class_idx in range(n_classes):
        df[f"class_{class_idx}_weighted"] = weighted_avg_probs[:, class_idx]

    # --- Evaluate accuracy metrics ---
    # Compute metrics for each group
    label_key = {0: "WT", 1: "FUS"}
    groups = ["agg", "weighted"] + antibodies
    results = {}
    
    for group in groups:
        cls_col = f'class_1_{group}' 
        preds = df.sort_values(by=cls_col, ascending=True)
        preds["True_label"] = preds["True"].map(label_key)
        
        # Compute metrics
        pred_labels = (preds[cls_col] > 0.5).map({True: "FUS", False: "WT"})
        accuracy = (pred_labels == preds["True_label"]).mean()
        
        # Cross entropy with clipped probabilities
        true_labels = preds["True"].values.astype(float)
        prob_class_1 = preds[cls_col].values.astype(float)
        eps = 1e-15
        prob_clipped = np.clip(prob_class_1, eps, 1 - eps)
        
        cross_entropy = -(
            true_labels * np.log(prob_clipped) + 
            (1 - true_labels) * np.log(1 - prob_clipped)
        ).mean()
        
        # AUC
        auc = roc_auc_score(true_labels.astype(int), prob_class_1)
        
        results[group] = {
            'accuracy': accuracy,
            'cross_entropy': cross_entropy,
            'auc': auc
        }
    
    return results, df


def run_stability_experiment(
    train_screen,
    test_screen,
    n_replicates=20,
    **kwargs
):
    """
    Run the multimarker binary analysis for multiple replicates.
    
    Parameters:
    -----------
    train_screen : ImageScreenMultiAntibody
        Preprocessed training screen object
    test_screen : ImageScreenMultiAntibody
        Preprocessed testing screen object
    antibodies : tuple
        Antibody markers to use
    n_replicates : int
        Number of replicates to run
    **kwargs : dict
        Additional arguments passed to run_multimarker_binary_analysis
        
    Returns:
    --------
    tuple
        (results_df, predictions_df) containing metrics and predictions
    """
    
    print("Starting replicate analysis...")
    all_results, all_predictions = [], []
    
    for replicate in range(n_replicates):
        print(f"Running replicate {replicate + 1}/{n_replicates}")
        
        results, predictions = run_multimarker_binary_analysis(
            train_screen=train_screen,
            test_screen=test_screen,
            antibodies=antibodies,
            random_seed=replicate, 
            **kwargs
        )
        
        # Add metadata to predictions
        predictions['replicate'] = replicate + 1
        predictions['random_seed'] = replicate
        for param, value in kwargs.items():
            predictions[param] = value
        all_predictions.append(predictions)
        
        # Convert results to long format
        for group, metrics in results.items():
            for metric, value in metrics.items():
                result_row = {
                    'replicate': replicate + 1,
                    'group': group,
                    'metric': metric,
                    'value': value,
                    'random_seed': replicate,
                    **kwargs  # Add all parameter values
                }
                all_results.append(result_row)
    
    # Create DataFrames 
    results_df = pd.DataFrame(all_results)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    return results_df, predictions_df


if __name__ == "__main__":
    from pathlib import Path
    
    # Grid search parameters
    n_layers_values = [0, 1, 2]
    d_model_values = [8, 16, 32]
    n_replicates = 10
    
    grid = {
        "n_layers": 0,
        "n_layers": 1,
        "n_layers": 2,
        "d_model": 8,
        "d_model": 16
    
    }
        
    all_results, all_predictions = [], []
    
    total_experiments = len(n_layers_values) * len(d_model_values) - 2 
    experiment_count = 0
    
    # Load in screen data 
    param_dir = "/home/kkumbier/als/scripts/maps/template_analyses/params/"
    train_params_path = Path(param_dir) / "maps_multiantibody-train.json"
    test_params_path = Path(param_dir) / "maps_multiantibody-test.json"
    antibodies = ["HSP70/SOD1", "FUS/EEA1"]
    
    with open(train_params_path, "r") as f:
        train_params = json.load(f)
    with open(test_params_path, "r") as f:
        test_params = json.load(f)

    train_screen = ImageScreenMultiAntibody(train_params)
    train_screen.load(antibody=antibodies)
    test_screen = ImageScreenMultiAntibody(test_params)
    test_screen.load(antibody=antibodies)

    print("Preprocessing training set...")
    train_screen.preprocess()
    print("Preprocessing test set...")
    test_screen.preprocess() 
    
    for n_layers in n_layers_values:
        for d_model in d_model_values:
            if n_layers == 0 and d_model != 16:
                continue
            
            experiment_count += 1
            print(f"\n=== Experiment {experiment_count}/{total_experiments}===")
            print(f"n_layers={n_layers}, d_model={d_model}")
            
            # Run stability experiment for this parameter combination
            results_df, predictions_df = run_stability_experiment(
                train_screen=train_screen,
                test_screen=test_screen,
                n_replicates=n_replicates,
                n_layers=n_layers,
                d_model=d_model
            ) 
            
            all_results.append(results_df)
            all_predictions.append(predictions_df)
    
    # Combine and save results
    print("\n=== Combining all results ===")
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    
    # Save results
    output_dir = Path("/home/kkumbier/maps/template_analyses/")
    combined_results.to_csv(output_dir / "multimarker_grid_search_results.csv", index=False)
    combined_predictions.to_csv(output_dir / "multimarker_grid_search_predictions.csv", index=False)
    
    print("Results saved successfully!")
    

