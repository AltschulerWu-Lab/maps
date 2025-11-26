"""
Test script to verify reproducibility of MAP model training pipeline.

This script trains models twice with the same seed and verifies that:
1. Predictions are identical
2. Model weights are consistent
3. Training is reproducible across runs

The script caches the preprocessed screen object to speed up subsequent runs.

Usage:
    python test_reproducibility.py
    python test_reproducibility.py --no-cache  # Skip loading cached screen
"""

import os
import sys
import random
import numpy as np
import torch
import json
import pickle
from pathlib import Path
import polars as pl

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from maps.screens import ImageScreenMultiAntibody
from maps.analyses import MAP


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable PyTorch deterministic algorithms (required for reproducibility)
    #torch.use_deterministic_algorithms(True, warn_only=True)


def create_minimal_params():
    """Create minimal parameter configuration for testing."""
    
    # Load base parameters
    pdir = Path("/home/kkumbier/maps/template_analyses/pipelines/params")
    with open(pdir / "multiclass.json", "r") as f:
        params = json.load(f)
    
    # Modify for minimal testing
    params["analysis"]["MAP"]["model"]["MultiAntibody"]["train"]["n_epochs"] = 3
    params["analysis"]["MAP"]["data_loader"]["n_cells"] = 10
    params["analysis"]["MAP"]["data_loader"]["batch_size"] = 4
    params["analysis"]["MAP"]["reps"] = 1
    
    return params


def load_or_create_screen(params, cache_file, use_cache=True):
    """Load screen from cache or create and cache it."""
    
    cache_path = Path(cache_file)
    
    if use_cache and cache_path.exists():
        print(f"Loading cached screen from {cache_file}...")
        with open(cache_path, "rb") as f:
            screen = pickle.load(f)
        print("Screen loaded from cache.")
        return screen
    
    print("Creating and preprocessing screen (this may take a minute)...")
    set_seed(42)  # Set seed before loading
    screen = ImageScreenMultiAntibody(params)
    screen.load(antibody=params["antibodies"])
    screen.preprocess()
    
    # Save to cache
    print(f"Saving screen to cache: {cache_file}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(screen, f)
    print("Screen cached successfully.")
    
    return screen


def run_training(seed, screen, diagnostics=False):
    """Run a single training iteration with given seed and preprocessed screen."""
    print(f"\n{'='*60}")
    print(f"Running training with seed: {seed}")
    print(f"{'='*60}\n")
    
    # Set seed before training
    set_seed(seed)
    
    diagnostics_data = {}
    
    if diagnostics:
        # Check data shape
        for ab in screen.data.keys():
            print(f"  {ab}: {screen.data[ab].shape}")
    
    # Train model
    print("Training model...")
    map_analysis = MAP(screen)
    map_analysis.fit()
    
    return (map_analysis, diagnostics_data) if diagnostics else map_analysis


def compare_predictions(pred1, pred2):
    """Compare two prediction DataFrames for equality."""
    
    if isinstance(pred1, dict):
        # Multi-mutation case
        keys1 = set(pred1.keys())
        keys2 = set(pred2.keys())
        if keys1 != keys2:
            print(f"❌ Different mutation keys: {keys1} vs {keys2}")
            return False
        
        all_match = True
        for key in keys1:
            match = compare_predictions(pred1[key], pred2[key])
            if not match:
                all_match = False
                print(f"❌ Predictions differ for mutation: {key}")
        
        return all_match
    
    # Single DataFrame case
    if not isinstance(pred1, pl.DataFrame) or not isinstance(pred2, pl.DataFrame):
        print(f"❌ Unexpected types: {type(pred1)}, {type(pred2)}")
        return False
    
    # Compare shapes
    if pred1.shape != pred2.shape:
        print(f"❌ Shape mismatch: {pred1.shape} vs {pred2.shape}")
        return False
    
    # Get prediction columns (those starting with 'Ypred' or 'Class_')
    pred_cols = [col for col in pred1.columns if col.startswith('Ypred') or col.startswith('Class_')]
    
    if not pred_cols:
        print("⚠️  No prediction columns found")
        return False
    
    # Compare predictions with tolerance
    tolerance = 1e-6
    all_close = True
    
    for col in pred_cols:
        vals1 = pred1[col].to_numpy()
        vals2 = pred2[col].to_numpy()
        
        max_diff = np.abs(vals1 - vals2).max()
        
        if not np.allclose(vals1, vals2, atol=tolerance):
            print(f"❌ Column {col} differs (max diff: {max_diff:.2e})")
            all_close = False
        else:
            print(f"✓ Column {col} matches (max diff: {max_diff:.2e})")
    
    return all_close


def check_dataloader_reproducibility(screen, params, seed):
    """Test if dataloaders produce the same batches."""
    from maps.multiantibody.data_loaders import create_multiantibody_dataloader
    
    print("\n" + "="*60)
    print("DATALOADER REPRODUCIBILITY CHECK")
    print("="*60)
    
    # Create dataloader config
    dl_config = params["analysis"]["MAP"]["data_loader"].copy()
    dl_config["mode"] = "train"
    dl_config["shuffle"] = True
    
    # First dataloader
    set_seed(seed)
    loader1 = create_multiantibody_dataloader(screen, **dl_config)
    batch1 = next(iter(loader1))
    
    # Second dataloader
    set_seed(seed)
    loader2 = create_multiantibody_dataloader(screen, **dl_config)
    batch2 = next(iter(loader2))
    
    # Compare batches
    print("\nComparing first batch from two dataloaders:")
    ab = list(batch1.keys())[0]
    
    # Compare labels
    labels1 = batch1[ab][1].numpy()
    labels2 = batch2[ab][1].numpy()
    labels_match = np.array_equal(labels1, labels2)
    print(f"  Labels match: {labels_match}")
    if not labels_match:
        print(f"    Run 1: {labels1}")
        print(f"    Run 2: {labels2}")
    
    # Compare data
    data1 = batch1[ab][0].numpy()
    data2 = batch2[ab][0].numpy()
    data_match = np.allclose(data1, data2, atol=1e-6)
    max_diff = np.abs(data1 - data2).max()
    print(f"  Data match: {data_match} (max diff: {max_diff:.2e})")
    
    return labels_match and data_match


def check_train_test_split(screen, seed):
    """Check if train/test splits are consistent."""
    from maps.fitter_utils import cellline_split
    
    print("\n" + "="*60)
    print("TRAIN/TEST SPLIT REPRODUCIBILITY CHECK")
    print("="*60)
    
    # Get splits twice
    set_seed(seed)
    split1 = cellline_split(screen, train_prop=0.5, type="CellLines")
    
    set_seed(seed)
    split2 = cellline_split(screen, train_prop=0.5, type="CellLines")
    
    # Compare
    train1 = split1["id_train"].sort("CellLines")["CellLines"].to_list()
    train2 = split2["id_train"].sort("CellLines")["CellLines"].to_list()
    
    test1 = split1["id_test"].sort("CellLines")["CellLines"].to_list()
    test2 = split2["id_test"].sort("CellLines")["CellLines"].to_list()
    
    train_match = train1 == train2
    test_match = test1 == test2
    
    print(f"\nTrain splits match: {train_match}")
    if not train_match:
        print(f"  Run 1: {train1}")
        print(f"  Run 2: {train2}")
    
    print(f"Test splits match: {test_match}")
    if not test_match:
        print(f"  Run 1: {test1}")
        print(f"  Run 2: {test2}")
    
    return train_match and test_match


def check_model_initialization(params, seed):
    """Check if model weights are initialized identically."""
    from maps.multiantibody.models import MultiAntibodyClassifier
    from maps.multiantibody.config import ModelConfig
    
    print("\n" + "="*60)
    print("MODEL INITIALIZATION REPRODUCIBILITY CHECK")
    print("="*60)
    
    model_config = ModelConfig(**params["analysis"]["MAP"]["model"]["MultiAntibody"]["model"])
    model_config.antibody_feature_dims = {"test": 100}
    
    # Initialize two models with same seed
    set_seed(seed)
    model1 = MultiAntibodyClassifier(**vars(model_config))
    
    set_seed(seed)
    model2 = MultiAntibodyClassifier(**vars(model_config))
    
    # Compare weights
    weights_match = True
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if not torch.allclose(param1, param2, atol=1e-6):
            print(f"  ❌ Weights differ in layer: {name1}")
            weights_match = False
            break
    
    if weights_match:
        print("  ✓ Model weights initialized identically")
    
    return weights_match


def main():
    """Main test function."""
    
    # Check for --no-cache flag
    use_cache = "--no-cache" not in sys.argv
    
    print("\n" + "="*60)
    print("REPRODUCIBILITY TEST FOR MAP PIPELINE")
    print("="*60)
    
    # Create minimal parameters
    params = create_minimal_params()
    
    print("\nTest Configuration:")
    print(f"  Screen: {params['screen']}")
    print(f"  Antibodies: {params['antibodies']}")
    print(f"  Epochs: {params['analysis']['MAP']['model']['MultiAntibody']['train']['n_epochs']}")
    print(f"  N cells: {params['analysis']['MAP']['data_loader']['n_cells']}")
    print(f"  Batch size: {params['analysis']['MAP']['data_loader']['batch_size']}")
    print(f"  Fitter: {params['analysis']['MAP']['fitter']}")
    print(f"  Use cache: {use_cache}")
    
    SEED = 42
    
    # Load or create screen
    cache_file = Path(__file__).parent / "cached_screen.pkl"
    screen = load_or_create_screen(params, cache_file, use_cache=use_cache)
    
    # Run diagnostic checks
    print("\n" + "="*60)
    print("RUNNING DIAGNOSTIC CHECKS")
    print("="*60)
    
    model_init_ok = check_model_initialization(params, SEED)
    split_ok = check_train_test_split(screen, SEED)
    dataloader_ok = check_dataloader_reproducibility(screen, params, SEED)
    
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    print(f"  Model initialization: {'✓' if model_init_ok else '❌'}")
    print(f"  Train/test splits: {'✓' if split_ok else '❌'}")
    print(f"  Dataloader batches: {'✓' if dataloader_ok else '❌'}")
    
    if not (model_init_ok and split_ok and dataloader_ok):
        print("\n❌ Basic reproducibility checks failed!")
        print("   Fix these issues before running full training.")
        return 1
    
    # Run training twice with same seed
    print("\n" + "="*60)
    print("FIRST TRAINING RUN")
    print("="*60)
    result1, diag1 = run_training(SEED, screen, diagnostics=True)
    
    print("\n" + "="*60)
    print("SECOND TRAINING RUN")
    print("="*60)
    result2, diag2 = run_training(SEED, screen, diagnostics=True)
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARING RESULTS")
    print("="*60 + "\n")
    
    predictions_match = compare_predictions(
        result1.fitted["predicted"], result2.fitted["predicted"]
    )
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    
    if predictions_match:
        print("\n✅ SUCCESS: Model training is reproducible!")
        print("   Predictions are identical across runs with the same seed.")
        return 0
    else:
        print("\n❌ FAILURE: Model training is NOT reproducible!")
        print("   Predictions differ between runs with the same seed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
