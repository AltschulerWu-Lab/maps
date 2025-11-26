""" Fitter functions are used to run classification pipelines used for MAP-scoring analysis. Fitters define data splits, how model(s) should be trained relative to these sample splits, how predictions should be generated relative to these splits.

All fitter functions operate on a `Screen` and `BaseModel` object. Some fitter functions additionally specify holdout cell lines—i.e., those that should be removed during training and only evaluated as model predictions.

Fitters return dictionaries containing the following keys: `fitted`: list of fitted models, `predicted`: DataFrame of predictions, `importance`: DataFrame of feature importances
    
`*_mut` fitters are wrappers that run an analysis for each mutational 
background. As a result, they return one dict per mutational background.
"""

import polars as pl
import pandas as pd
import numpy as np

from typing import Dict
from maps.models import BaseModel
from maps.fitter_utils import (
    cellline_split, 
    fit_split, 
    get_mutation_celllines, 
    merge_metadata
)

from maps.multiantibody.config import DataLoaderConfig
from maps.models import SKLearnModel, PyTorchModel

import torch
from typing import TYPE_CHECKING, List, Optional
import copy

if TYPE_CHECKING:
    from maps.screens import ScreenBase
    from maps.screens import ImageScreenMultiAntibody

# --- Leave one out training loops ---
def leave_one_out_mut(screen: 'ScreenBase', model: BaseModel) -> Dict:
    """Wrapper for running leave-one-out fitter by mutational background.
    
    Binary classifiers for <mutation> vs WT will be run for all ALS mutational 
    backgrounds, excluding sporadics. Each mutation is trained separately using
    leave-one-out cross-validation on cell lines.
    
    Args:
        screen (ScreenBase): Screen object containing data and metadata.
        model (BaseModel): Model instance (SKLearnModel or PyTorchModel).
        
    Returns:
        Dict[str, Dict]: Dictionary keyed by mutation name, each containing:
            - `fitted` (list): List of fitted models
            - `predicted` (pl.DataFrame): Predictions with metadata
            - `importance` (pd.DataFrame or None): Feature importances
            - Additional keys depending on model type
    """
    assert screen.data is not None, "screen data not loaded"
    assert screen.metadata is not None, "screen metadata not loaded"
    
    metadata = merge_metadata(screen)
    mutations = metadata["Mutations"].unique().sort()
    mutations = set(mutations) - set(["WT", "sporadic"])
    out = {}
    
    for m in mutations:
        print(f"Training {m}...")
        mutations_holdout = list(mutations - {m}) + ["sporadic"]
        holdout = get_mutation_celllines(metadata, mutations_holdout)
        out[m] = leave_one_out(screen, model, holdout)
         
    return out

def leave_one_out(screen: 'ScreenBase', model: BaseModel, holdout: List = []) -> Dict:
    """Wrapper for leave-one-out cross-validation.
    
    Dispatches to sklearn or pytorch implementation based on model type. Leaves
    one cell line out at a time, trains on remaining cell lines, and predicts
    on the held-out cell line plus any additional holdout cell lines specified.
    
    Args:
        screen (ScreenBase): Screen object containing data and metadata.
        model (BaseModel): Model instance (SKLearnModel or PyTorchModel).
        holdout (List[str], optional): Additional cell lines to hold out from 
            training. Defaults to [].
            
    Returns:
        Dict: Dictionary containing:
            - `fitted` (list): List of fitted models (one per cell line)
            - `predicted` (pl.DataFrame): Predictions with metadata
            - `importance` (pd.DataFrame or None): Feature importances
            - Additional keys for PyTorch models (scalers, training_lines)
    """
    metadata = merge_metadata(screen)
    holdout += get_mutation_celllines(metadata, ["sporadic"])
    
    if isinstance(model, SKLearnModel):
        return leave_one_out_sklearn(screen, model, holdout)
    elif isinstance(model, PyTorchModel):
        return leave_one_out_pytorch(screen, model, holdout)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def leave_one_out_sklearn(
    screen: "ScreenBase", 
    model: BaseModel, 
    holdout: List = []) -> Dict:
    """Leave-one-out cross-validation for scikit-learn models.
    
    Iterates through each cell line, training on all other cell lines and 
    predicting on the held-out cell line. Models are trained on well-averaged
    features.
    
    Args:
        screen (ScreenBase): Screen object with data and metadata.
        model (BaseModel): SKLearnModel instance.
        holdout (List[str], optional): Cell lines to exclude from training.
            Defaults to [].
            
    Returns:
        Dict: Dictionary containing:
            - `fitted` (list): List of fitted model copies
            - `predicted` (pl.DataFrame): Predictions joined with metadata
            - `importance` (pd.DataFrame): Feature importances from each fold
    """
    assert screen.data is not None, "screen data not loaded"
    assert screen.metadata is not None, "screen metadata not loaded"
    y = screen.get_response() 
    x = screen.get_data()
    
    # Get unique cell lines and exclude test cell lines
    cell_lines = (
        screen.metadata
        .select("CellLines")
        .unique()
        .sort("CellLines")
        .to_series()
        .to_list()
    )
    
    cell_lines = set(cell_lines) - set(holdout)
    
    fitted, predicted, importance = [], [], []
     
    for cl in cell_lines: 
        cell_lines_test = [cl]
        cell_lines_test = cell_lines_test + holdout
        
        # Create train and test indices
        id_train = screen.metadata \
            .filter(~pl.col("CellLines").is_in(cell_lines_test)) \
            .select("ID") \
            .to_series()

        id_test = screen.metadata \
            .filter(pl.col("CellLines").is_in(cell_lines_test)) \
            .select("ID") \
            .to_series()
        
        # Fit model and make predictions
        model.fit(x=x, y=y, id_train=id_train)
        predicted_cl = model.predict(x=x, id_test=id_test)
        predicted_cl= predicted_cl.with_columns(pl.lit(cl).alias("Holdout"))
        
        # Merge results
        predicted.append(predicted_cl.join(screen.metadata, on="ID"))
        fitted.append(copy.deepcopy(model))
        importance.append(model.get_importance(x))
      
    out = {}  
    out["fitted"] = fitted
    out["predicted"] = pl.concat(predicted)
    out["importance"] = pd.concat([pd.Series(p) for p in importance], axis=1)
    return out
        

def leave_one_out_pytorch(
    screen: "ScreenBase", 
    model: BaseModel, 
    holdout: List = []) -> Dict:
    """Leave-one-cell-line-out classification for PyTorch models.
    
    Trains PyTorch models using DataLoader batching on single-cell data.
    Each iteration holds out one cell line for testing while training on all
    remaining cell lines.
    
    Args:
        screen (ScreenBase): Screen object (ImageScreen or ImageScreenMultiAntibody).
        model (BaseModel): PyTorchModel instance.
        holdout (List[str], optional): Cell lines to exclude from training.
            Defaults to [].
            
    Returns:
        Dict: Dictionary containing:
            - `fitted` (list): List of fitted model copies (moved to CPU)
            - `predicted` (pl.DataFrame): Concatenated predictions
            - `scalers` (list): Feature scalers for each fold
            - `training_lines` (list): Cell lines used in training
            - `importance` (None): Not implemented for PyTorch models
    """
    assert screen.data is not None, "screen data not loaded"
    assert screen.metadata is not None, "screen metadata not loaded"

    map_params = screen.params.get("analysis").get("MAP", None)
    assert map_params is not None, "MAP parameters not found"
    
    # Set configs for batch loading 
    dataloader_config = DataLoaderConfig(**map_params.get("data_loader", {}))

    # Get all cell lines
    if isinstance(screen.metadata, dict):
        cell_lines = [s["CellLines"].unique().sort() for s in screen.metadata.values()]
        cell_lines = pl.concat(cell_lines).unique().sort().to_list()
    else:
        cell_lines = screen.metadata["CellLines"].unique().sort().to_list()
    cell_lines = list(set(cell_lines) - set(holdout))

    fitted, predicted, scalers_list, split = [], [], [], []

    for cl in cell_lines:
        # Define test set: current cell line + holdout
        print(f"---Training hold-out: {cl}---")
        cell_lines_test = set([cl] + holdout)
        train_celllines = list(set(cell_lines) - cell_lines_test)
        split.append(train_celllines)
        
        # Train and predict for current holdout cell line
        fit_model, ypred, scalers = fit_split(
            screen, model, train_celllines, dataloader_config
        )

        predicted.append(ypred)
        fitted_model = getattr(fit_model, 'model', fit_model)

        if isinstance(fitted_model, torch.nn.Module):
            fitted.append(copy.deepcopy(fitted_model.to("cpu")))
        else:
            fitted.append(copy.deepcopy(fitted_model))
            
        scalers_list.append(scalers)

    # Use Polars concat for predictions
    out = {
        "fitted": fitted,
        "predicted": pl.concat(predicted),
        "scalers": scalers_list,
        "training_lines": cell_lines,
        "importance": None
    }
    
    return out

# --- Sample split training loops ---
def sample_split_mut(screen: 'ScreenBase', model: BaseModel) -> Dict:
    """Wrapper for running sample_split fitter by mutational background.
    
    Binary classifiers for <mutation> vs WT will be run for all ALS mutational 
    backgrounds, excluding sporadics. For each mutation, performs a 50/50 sample
    split with reciprocal training and prediction.
    
    Args:
        screen (ScreenBase): Screen object containing data and metadata.
        model (BaseModel): Model instance (SKLearnModel or PyTorchModel).
        
    Returns:
        Dict[str, Dict]: Dictionary keyed by mutation name, each containing:
            - `fitted` (list): List of fitted models (2 per mutation)
            - `predicted` (pl.DataFrame): Predictions with metadata
            - `training_lines` (list): Cell lines in each training split
            - Additional keys depending on model type
    """
    assert screen.data is not None, "screen data not loaded"
    assert screen.metadata is not None, "screen metadata not loaded"

    metadata = merge_metadata(screen)
    mutations = metadata["Mutations"].unique().sort()
    mutations = set(mutations) - set(["WT", "sporadic"])
    
    out = {}
    for m in mutations:
        print(f"Training {m}...")
        mutations_holdout = list(mutations - {m}) + ["sporadic"]
        holdout = get_mutation_celllines(metadata, mutations_holdout)
        out[m] = sample_split(screen, model, holdout)
        
    return out


def sample_split(screen: 'ScreenBase', model: BaseModel, holdout: List = []) -> Dict:
    """Wrapper for sample split cross-validation with hold-out cell lines.
    
    Splits cell lines 50/50, trains two models reciprocally (each on one split,
    predicting on the other), and combines predictions. Supports multiple 
    replicates via params configuration.
    
    Args:
        screen (ScreenBase): Screen object with data and metadata.
        model (BaseModel): Model instance (SKLearnModel or PyTorchModel).
        holdout (List[str], optional): Cell lines to exclude from training.
            Defaults to [].
        seed (int, optional): Random seed for reproducibility. If None, will check
            if numpy seed has been set and use that; otherwise defaults to 47.
        
    Returns:
        Dict: Dictionary containing:
            - `fitted` (list): List of all fitted models from all replicates
            - `predicted` (pl.DataFrame): Concatenated predictions from replicates
            - `training_lines` (list): Cell lines in each training split
            - Additional keys depending on model type
    """
    metadata = merge_metadata(screen)
    holdout += get_mutation_celllines(metadata, ["sporadic"])

    reps = screen.params.get("analysis").get("MAP", {}).get("reps", 1)
    out_list = []
    for i in range(reps):
        print(f"--- Replicate {i+1}/{reps} ---")
        if isinstance(model, SKLearnModel):
            out_list.append(sample_split_sklearn(screen, model, holdout))
        elif isinstance(model, PyTorchModel):
            out_list.append(sample_split_pytorch(screen, model, holdout))
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

    # Reformat output list into dict matching sample_split_* structure
    keys = out_list[0].keys()
    out = {}
    for k in keys:
        if k == 'predicted':
            dfs = [d[k] for d in out_list if d[k] is not None]
            out[k] = pl.concat(dfs)
        elif k == 'importance':
            out[k] = None
        else:
            vals = []
            for d in out_list:
                vals.extend(d[k])
            out[k] = vals
    
    return out


def sample_split_pytorch(
    screen: "ScreenBase", 
    model: BaseModel, 
    holdout: List = []) -> Dict:
    """Sample split cross-validation for PyTorch models.
    
    Performs 50/50 cell line split with reciprocal training using DataLoader
    batching on single-cell data. Each model is trained on one split and 
    predicts on the complementary split plus holdout cell lines.
    
    Args:
        screen (ScreenBase): Screen object (ImageScreen or ImageScreenMultiAntibody).
        model (BaseModel): PyTorchModel instance.
        holdout (List[str], optional): Cell lines to exclude from training.
            Defaults to [].
        seed (int, optional): Random seed for split generation. Defaults to 47.
        
    Returns:
        Dict: Dictionary containing:
            - `fitted` (list): Two fitted models (moved to CPU)
            - `predicted` (pl.DataFrame): Concatenated predictions
            - `training_lines` (list): Cell lines in each training split (2 lists)
            - `scalers` (list): Feature scalers for each split (2 scalers)
            - `importance` (None): Not implemented for PyTorch models
    """
    assert screen.data is not None, "screen data not loaded"
    assert screen.metadata is not None, "screen metadata not loaded"
    assert isinstance(model, PyTorchModel), "model must be a PyTorchModel"
    
    map_params = screen.params.get("analysis").get("MAP", None)
    assert map_params is not None, "MAP parameters not found"
    dataloader_config = DataLoaderConfig(**map_params.get("data_loader", {}))
    
    # Define 50/50 sample split by mutation using cellline_split
    split = cellline_split(screen, train_prop=0.5, type="CellLines")
    split1_celllines = split["id_train"].to_series()
    split2_celllines = split["id_test"].to_series()
    
    # Remove holdout cell lines from training sets
    split1_celllines = list(set(split1_celllines) - set(holdout))
    split2_celllines = list(set(split2_celllines) - set(holdout))

    # Train on split1, predict on everything except split1
    fit1, yp1, scalers1 = fit_split(
        screen, model, split1_celllines, dataloader_config
    )

    # Train on split2, predict on everything except split2
    fit2, yp2, scalers2 = fit_split(
        screen, model, split2_celllines, dataloader_config
    )

    # Use Polars concat for predictions
    predicted = pl.concat([yp1, yp2])

    # Safely handle .to('cpu') if available
    fitted_model1 = getattr(fit1, 'model', fit1)
    fitted_model2 = getattr(fit2, 'model', fit2)

    if isinstance(fitted_model1, torch.nn.Module):
        fitted1 = copy.deepcopy(fitted_model1.to("cpu"))
    else:
        fitted1 = copy.deepcopy(fitted_model1)
        
    if isinstance(fitted_model2, torch.nn.Module):
        fitted2 = copy.deepcopy(fitted_model2.to("cpu"))
    else:
        fitted2 = copy.deepcopy(fitted_model2)
    
    out = {
        'fitted': [fitted1, fitted2],
        'predicted': predicted,
        'training_lines': [split1_celllines, split2_celllines],
        'scalers': [scalers1, scalers2],
        'importance': None
    }

    return out
    
    
def sample_split_sklearn(
    screen: 'ScreenBase', 
    model: BaseModel, 
    holdout: List = []) -> Dict:
    """Sample split cross-validation for sklearn models.
    
    Performs 50/50 cell line split with reciprocal training on well-averaged
    features. Each model is trained on one split and predicts on the 
    complementary split. Predictions are centered around 0.5.
    
    Args:
        screen (ScreenBase): Screen object with data and metadata.
        model (BaseModel): SKLearnModel instance.
        holdout (List[str], optional): Cell lines to exclude from training.
            Defaults to [].
        seed (int, optional): Random seed for split generation. Defaults to 47.
        
    Returns:
        Dict: Dictionary containing:
            - `fitted` (list): Two fitted models
            - `predicted` (pl.DataFrame): Predictions joined with metadata, centered
            - `importance` (pd.Series or None): Average feature importances
    """
    assert screen.data is not None, "screen data not loaded"
    assert screen.metadata is not None, "screen metadata not loaded"
    
    out = {}
    
    # Define 50/50 sample split by mutation and set sporadics as hold-out
    split = cellline_split(screen, train_prop=0.5, type="CellLines")

    if holdout is not None:
        id_holdout = screen.metadata \
            .filter(pl.col("CellLines").is_in(holdout)) \
            .select("ID") \
            .to_series()
    else:
        id_holdout = pl.Series([])
             
    # Initialize data for model fitting
    y = screen.get_response() 
    x = screen.get_data()
   
    # Fit models to each sample split
    id1 = split["id_train"].filter(~split["id_train"].is_in(id_holdout))
    fit1 = model.fit(x=x, y=y, id_train=id1)
    
    id2 = split["id_test"].filter(~split["id_test"].is_in(id_holdout))
    fit2 = model.fit(x=x, y=y, id_train=id2)

    out['fitted'] = [fit1, fit2]
    
    # Generate model predictions
    id2 = pl.concat([split["id_test"], id_holdout]).unique()
    yp2 = model.predict(fit1, x, id_test=id2)  
    yp2 = yp2.with_columns((pl.col("Ypred") - pl.col("Ypred").mean() +  0.5))
    
    id1 = pl.concat([split["id_train"], id_holdout]).unique() 
    yp1 = model.predict(fit2, x, id_test=split['id_train'])
    yp1 = yp1.with_columns((pl.col("Ypred") - pl.col("Ypred").mean() +  0.5))

    out['predicted'] = pl.concat([yp1, yp2]).join(screen.metadata, on="ID")
    
    # Generate model feature importance
    imp1 = model.get_importance(x)
    imp2 = model.get_importance(x)
    if imp1 is not None and imp2 is not None:
        out['importance'] = (imp1 + imp2) / 2
    else:
        out['importance'] = None
        
    return out