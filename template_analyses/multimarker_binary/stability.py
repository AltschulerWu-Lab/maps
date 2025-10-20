import pandas as pd
import numpy as np
import json
import torch
import sys
from pathlib import Path
from sklearn.metrics import roc_auc_score

# TODO: class indices are flipping randomly for replicate runs rather than 
# using values specified in response_map

# NOTE: This script follows the sample split strategy from baseline_classification.ipynb:
# 1. Train model_train on train_screen, evaluate on test_screen  
# 2. Train model_test on test_screen, evaluate on train_screen
# 3. Concatenate predictions from both models for final evaluation

# Add maps to path if needed
sys.path.append("/home/kkumbier/maps/")

from maps.screens import ImageScreenMultiAntibody
from maps.multiantibody.config import TrainConfig, ModelConfig, DataLoaderConfig
import maps.multiantibody.data_loaders as data_loaders
import maps.multiantibody.models as models
import maps.multiantibody.training as training
import maps.multiantibody.evaluate as evaluate

def run_multimarker_binary_analysis(
    train_screen,
    test_screen,
    response_map={"WT": 0, "FUS": 1},
    n_classes=2,
    batch_size=6,
    n_epochs=50,
    patience=10,
    lr=5e-3,
    d_model=16,
    n_layers=0,
    use_contrastive_loss=False,
    random_seed=None
):
    """
    Run the multimarker binary classification analysis using sample split strategy.
    
    This follows the notebook pattern:
    1. Train model_train on train_screen, evaluate on test_screen
    2. Train model_test on test_screen, evaluate on train_screen  
    3. Concatenate predictions from both models for final evaluation
    
    Parameters:
    -----------
    train_screen : ImageScreenMultiAntibody
        Preprocessed training screen
    test_screen : ImageScreenMultiAntibody
        Preprocessed test screen
    response_map : dict
        Mapping from response strings to numeric labels
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
    use_contrastive_loss: bool
        Whether to use contrastive loss
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
    dataloader_config.response_map = {"Mutations": response_map}

    train_dataloader = data_loaders.create_multiantibody_dataloader(
        train_screen, **vars(dataloader_config)
    )

    scalers = train_dataloader._get_scalers()
    test_dataloader = data_loaders.create_multiantibody_dataloader(
        test_screen, scalers=scalers, **vars(dataloader_config)
    )

    # Configure training parameters
    train_config.n_epochs = n_epochs
    train_config.patience = patience
    train_config.lr = lr
    train_config.use_contrastive_loss = use_contrastive_loss

    model_config.antibody_feature_dims = train_dataloader._get_feature_dims()
    model_config.d_model = d_model
    model_config.n_layers = n_layers

    # --- Training and Evaluation (following notebook pattern) ---
    
    # Model 1: Train on train_screen, evaluate on test_screen
    model_train = models.MultiAntibodyClassifier(**vars(model_config))
    training.train(model_train, train_dataloader, train_config)
    
    # Evaluate model_train on test data
    test_dataloader.mode = "eval"
    df_test, _ = evaluate.eval(model_train, test_dataloader)

    # Model 2: Train on test_screen, evaluate on train_screen
    model_test = models.MultiAntibodyClassifier(**vars(model_config))
    test_dataloader.mode = "train"
    training.train(model_test, test_dataloader, train_config)

    # Evaluate model_test on train data
    train_dataloader.mode = "eval"
    df_train, _ = evaluate.eval(model_test, train_dataloader)

    # --- Concatenate predictions (following notebook pattern) ---
    df = pd.concat([df_train, df_test], ignore_index=True)

    # --- Evaluate accuracy metrics on concatenated predictions ---
    groups = ["agg", "entropy"] + model_train.antibodies
    results = {}
    
    for group in groups:
        cls_col = f'class_1_{group}' 
        preds = df.sort_values(by=cls_col, ascending=True)
        
        # Compute metrics
        pred_labels = (preds[cls_col] > 0.5).map({True: 1, False: 0})
        accuracy = (pred_labels == preds["True"]).mean()
        
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
    Wrapper for running the multimarker binary analysis for multiple replicates.
    """
    
    print("Starting replicate analysis...")
    all_results, all_predictions = [], []
    
    for replicate in range(n_replicates):
        print(f"Running replicate {replicate + 1}/{n_replicates}")
        
        results, predictions = run_multimarker_binary_analysis(
            train_screen=train_screen,
            test_screen=test_screen,
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

    params = [
        [0, 16, False],
        [1, 8, False],
        [2, 8, False],
        [1, 16, False],
        [2, 16, False],
        [1, 32, False],
        [2, 32, False],
        [1, 16, True]
    ]

    cols = ["n_layers", "d_model", "use_contrastive_loss"]
    grid = pd.DataFrame(params, columns=cols)


    all_results, all_predictions = [], []    
    total_experiments = len(grid)
    experiment_count = 0
    n_replicates = 50
    
    # Load in screen data 
    param_dir = "/home/kkumbier/als/scripts/maps/template_analyses/params/"
    train_params_path = Path(param_dir) / "binary-split-train.json"
    test_params_path = Path(param_dir) / "binary-split-test.json"
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
    
    for idx, row in grid.iterrows():
        n_layers = row['n_layers']
        d_model = row['d_model']
        use_contrastive_loss = row['use_contrastive_loss']

        experiment_count += 1
        print(f"\n=== Experiment {experiment_count}/{len(grid)}===")

        # Run stability experiment for this parameter combination
        results_df, predictions_df = run_stability_experiment(
            train_screen=train_screen,
            test_screen=test_screen,
            n_replicates=n_replicates,
            n_layers=n_layers,
            d_model=d_model,
            use_contrastive_loss=use_contrastive_loss
        )
        
        all_results.append(results_df)
        all_predictions.append(predictions_df)
    
    # Combine and save results
    print("\n=== Combining all results ===")
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    
    # Save results
    output_dir = Path(
        "/home/kkumbier/maps/template_analyses/multimarker_binary"
    )
    combined_results.to_csv(output_dir / "binary_stability.csv", index=False)
    combined_predictions.to_csv(
        output_dir / "binary_predictions.csv", index=False
    )
    
    print("Results saved successfully!")
    

