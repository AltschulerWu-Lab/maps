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
    antibodies=["HSP70/SOD1", "FUS/EEA1"],
    n_classes=2,
    batch_size=6,
    n_epochs=100,
    patience=50,
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
    antibodies : list
        List of antibody markers to use
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

    # --- Model and training setup ---
    # Initialize configs
    dataloader_config = DataLoaderConfig()
    train_config = TrainConfig()
    model_config = ModelConfig()
    model_config.n_classes = n_classes

    # Create dataloaders
    dataloader_config.batch_size = batch_size
    train_dataloader = data_loaders.create_multiantibody_dataloader(
        train_screen,
        **vars(dataloader_config)
    )

    scalers = train_dataloader._get_scalers()
    dataloader_config.mode = "eval"
    test_dataloader = data_loaders.create_multiantibody_dataloader(
        test_screen,
        scalers=scalers,
        **vars(dataloader_config)
    )

    # Configure and train model
    train_config.n_epochs = n_epochs
    train_config.patience = patience
    train_config.lr = lr

    model_config.antibody_feature_dims = train_dataloader._get_feature_dims()
    model_config.d_model = d_model
    model_config.n_layers = n_layers

    model = models.MultiAntibodyClassifier(**vars(model_config))
    training.train(model, train_dataloader, train_config)

    # --- Evaluation ---
    model.eval()
    probs_line = []
    probs_cell = {}
    labels = []
    cell_lines = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in test_dataloader:
            if batch is None:
                continue
            
            x_dict = {ab: batch[ab][0].to(device) for ab in batch}
            y_line = batch[list(batch.keys())[0]][1].to(device)
            cell_lines.extend([batch[list(batch.keys())[0]][-1]])

            cell_logits, line_logits, embs = model(
                x_dict, return_embedding=True
            )        
            
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

    # Create DataFrame with predictions
    df = pd.DataFrame({
        "class_0_agg": probs_line[:, 0].numpy(),
        "class_1_agg": probs_line[:, 1].numpy()
    })

    # Add per-antibody class probabilities
    for ab in probs_cell:
        df[f"class_0_{ab}"] = probs_cell[ab][:, 0].numpy()
        df[f"class_1_{ab}"] = probs_cell[ab][:, 1].numpy()

    df["CellLines"] = cell_lines
    df["True"] = labels.numpy()

    # Compute metrics for each group
    label_key = {0: "WT", 1: "FUS"}
    groups = ["agg"] + antibodies
   
    # --- Generate result table --- 
    results = {}
    
    for g in groups:
        cls = f'class_1_{g}' 
        preds = df.sort_values(by=cls, ascending=True)
        preds["True_label"] = preds["True"].map(label_key)
        
        # Compute prediction accuracy
        pred_labels = (preds[cls] > 0.5).map({True: "FUS", False: "WT"})
        accuracy = (pred_labels == preds["True_label"]).mean()
        
        # Compute cross entropy, w/ clipped probabilities to avoid log(0)
        true_labels = preds["True"].values
        prob_class_1 = preds[cls].values

        eps = 1e-15
        prob_class_1_clipped = np.clip(prob_class_1.astype(float), eps, 1 - eps)
        
        cross_entropy = -(true_labels.astype(float) * np.log(prob_class_1_clipped) + 
                         (1 - true_labels.astype(float)) * np.log(1 - prob_class_1_clipped)).mean()
        
        # Compute AUC
        auc = roc_auc_score(true_labels.astype(int), prob_class_1.astype(float))
        
        results[g] = {
            'accuracy': accuracy,
            'cross_entropy': cross_entropy,
            'auc': auc
        }
    
    return results, df


def run_stability_experiment(
    train_params_path="/home/kkumbier/als/scripts/maps/template_analyses/params/maps_multiantibody-train.json",
    test_params_path="/home/kkumbier/als/scripts/maps/template_analyses/params/maps_multiantibody-test.json",
    antibodies=["HSP70/SOD1", "FUS/EEA1"],
    n_replicates=20,
    **kwargs
):
    """
    Run the multimarker binary analysis for multiple replicates and return results.
    
    Parameters:
    -----------
    train_params_path : str
        Path to training parameters JSON file
    test_params_path : str
        Path to testing parameters JSON file
    antibodies : list
        List of antibody markers to use
    n_replicates : int
        Number of replicates to run
    **kwargs : dict
        Additional arguments passed to run_multimarker_binary_analysis
        
    Returns:
    --------
    tuple
        (results_df, predictions_df) containing metrics and predictions
    """
    
    # Load and preprocess screens once for all replicates
    print("Loading and preprocessing screens...")
    
    # Load parameters
    with open(train_params_path, "r") as f:
        train_params = json.load(f)
        
    with open(test_params_path, "r") as f:
        test_params = json.load(f)

    # Load and process screens
    train_screen = ImageScreenMultiAntibody(train_params)
    train_screen.load(antibody=antibodies)

    test_screen = ImageScreenMultiAntibody(test_params)
    test_screen.load(antibody=antibodies)

    print("Preprocessing training set...")
    train_screen.preprocess()
    print("Preprocessing test set...")
    test_screen.preprocess()
    
    print("Starting replicate analysis...")
    
    all_results = []
    all_predictions = []
    
    for replicate in range(n_replicates):
        print(f"Running replicate {replicate + 1}/{n_replicates}")
        
        # Use different random seed for each replicate
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
                    'random_seed': replicate
                }
                # Add parameter values to results
                for param, param_value in kwargs.items():
                    result_row[param] = param_value
                all_results.append(result_row)
    
    # Create DataFrames 
    results_df = pd.DataFrame(all_results)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    
    # Print summary statistics for this parameter combination
    print("\nSummary Statistics:")
    summary = results_df.groupby(['group', 'metric'])['value'].agg(['mean', 'std', 'min', 'max'])
    print(summary)
    
    return results_df, predictions_df


if __name__ == "__main__":
    # Grid search parameters
    n_layers_values = [0, 1, 2]
    d_model_values = [8, 16, 32]
    n_replicates = 20
    
    all_results = []
    all_predictions = []
    
    total_experiments = len(n_layers_values) * len(d_model_values)
    experiment_count = 0
    
    for n_layers in n_layers_values:
        for d_model in d_model_values:
            experiment_count += 1
            print(f"\n=== Experiment {experiment_count}/{total_experiments}: n_layers={n_layers}, d_model={d_model} ===")
            
            # Run stability experiment for this parameter combination
            results_df, predictions_df = run_stability_experiment(
                n_replicates=n_replicates,
                n_layers=n_layers,
                d_model=d_model
            ) 
            
            all_results.append(results_df)
            all_predictions.append(predictions_df)
    
    # Combine all results
    print("\n=== Combining all results ===")
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    
    # Save combined results
    combined_results.to_csv("/home/kkumbier/maps/template_analyses/multimarker_grid_search_results.csv", index=False)
    combined_predictions.to_csv("/home/kkumbier/maps/template_analyses/multimarker_grid_search_predictions.csv", index=False)
    print("DONE!")
    
    # Print overall summary
    print("\n=== Grid Search Summary ===")
    summary_by_params = combined_results.groupby(['n_layers', 'd_model', 'group', 'metric'])['value'].agg(['mean', 'std']).round(4)
    print(summary_by_params)
