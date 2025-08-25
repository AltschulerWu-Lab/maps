""" Fitter functions are used to run classification pipelines used for MAP-scoring analysis. Fitters define data splits, how model(s) should be trained relative to these sample splits, how predictions should be generated relative to these splits.

All fitter functions operate on a `Screen` and `BaseModel` object. Some fitter functions additionally specify holdout cell lines—i.e., those that should be removed during training and only evaluated as model predictions.

Fitters return dictionaries containing the following keys: `fitted`: list of fitted models, `predicted`: DataFrame of predictions, `importance`: DataFrame of feature importances
    
`*_mut` fitters are wrappers that run an analysis for each mutational 
background. As a result, they return one dict per mutational background.
"""

import polars as pl
import pandas as pd

from typing import Dict
from maps.models import BaseModel
from maps.processing import select_sample_by_feature
from maps.fitter_utils import cellline_split

from maps.multiantibody.config import DataLoaderConfig
from maps.models import SKLearnModel, PyTorchModel

from typing import TYPE_CHECKING, List
import copy

if TYPE_CHECKING:
    from maps.screens import ScreenBase
    from maps.screens import ImageScreenMultiAntibody

# --- Leave one out training loops ---
def leave_one_out_mut(screen: 'ScreenBase', model: BaseModel) -> Dict:
    """ Wapper for running `leave_one_out` fitter by mutational background. Binary classifiers for <mutation> vs WT will be run for all ALS mutational backgrounds, excluding sporadics—which receive special treatment in sample_split.
    """
    assert screen.data is not None, "screen data not loaded"
    assert screen.metadata is not None, "screen metadata not loaded"
    
    mutations = screen.metadata["Mutations"].unique()
    mutations = set(mutations) - set(["WT", "sporadic"])
    out = {}
    
    for m in mutations:
        print(f"Training {m}...") 
        
        # Set sALS cell lines as holdouts
        mutations_holdout = list(mutations - {m}) + ["sporadic"]
        
        holdout = screen.metadata \
            .filter(pl.col("Mutations").is_in(mutations_holdout)) \
            .select("CellLines") \
            .to_series() \
            .to_list()
            
        out[m] = leave_one_out(screen, model, holdout)
         
    return out

def leave_one_out(screen, model, holdout=None):
    """Wrapper for leave-one-out cross-validation with hold-out cell lines. Dispatches to sklearn or pytorch implementation based on model type."""
    if isinstance(model, SKLearnModel):
        return leave_one_out_sklearn(screen, model, holdout)
    elif isinstance(model, PyTorchModel):
        return leave_one_out_pytorch(screen, model, holdout)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def leave_one_out_sklearn(
    screen: "ScreenBase", 
    model: BaseModel, 
    holdout: List|None=None):
    """Handles leave-one-out cross-validation for scikit-learn models, which take dataframe arguments for fitting and prediction.
    """
    assert screen.data is not None, "screen data not loaded"
    assert screen.metadata is not None, "screen metadata not loaded"
    y = screen.get_response() 
    x = screen.get_data()
    
    # Get unique cell lines and exclude test cell lines
    cell_lines = screen.metadata \
        .select("CellLines") \
        .unique() \
        .to_series() \
        .to_list()
    
    if holdout is not None:
        cell_lines = set(cell_lines) - set(holdout)
    
    fitted = []
    predicted = []
    importance = []
     
    for cl in cell_lines: 
        cell_lines_test = [cl]
       
        if holdout is not None:
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
    screen: "ImageScreenMultiAntibody", 
    model: BaseModel, 
    holdout: List|None=None
):
    """ Handles leave-one-cell-line-out classification for PyTorch models, which take a DataLoader for training and prediction.
    """
    assert screen.data is not None, "screen data not loaded"
    assert screen.metadata is not None, "screen metadata not loaded"
    assert isinstance(screen.data, dict), "expected screen dictionary"
    assert isinstance(screen.metadata, dict), "expected screen metadata dictionary"

    map_params = screen.params.get("analysis").get("MAP", None)
    assert map_params is not None, "MAP parameters not found"
    
    # Set configs for batch loading 
    dataloader_config = DataLoaderConfig(**map_params.get("data_loader", {}))

    # Initialize training set of cell lines
    cell_lines = [s["CellLines"].unique() for s in screen.metadata.values()]
    cell_lines = pl.concat(cell_lines).to_list()
    if holdout is not None:
        cell_lines = list(set(cell_lines) - set(holdout))

    # --- Iterate over each training cell line and fit model ---
    fitted = []
    predicted = []
    for cl in cell_lines:
        cell_lines_test = [cl]
        if holdout is not None:
            cell_lines_test += holdout
        cell_lines_test = pl.Series(cell_lines_test).implode()

        # Split metadata for train/test for current cell line
        train_screen = copy.deepcopy(screen)
        test_screen = copy.deepcopy(screen)
        
        for ab in screen.data.keys():
            train_screen.metadata[ab] = screen.metadata[ab].filter(
                ~pl.col("CellLines").is_in(cell_lines_test)
            )
            
            test_screen.metadata[ab] = screen.metadata[ab].filter(
                pl.col("CellLines").is_in(cell_lines_test)
            )
            
            train_screen.data[ab] = screen.data[ab].filter(
                pl.col("ID").is_in(train_screen.metadata[ab]["ID"].implode())
            )

            test_screen.data[ab] = screen.data[ab].filter(
                pl.col("ID").is_in(test_screen.metadata[ab]["ID"].implode())
            )

        # Create dataloaders
        from maps.multiantibody.data_loaders import create_multiantibody_dataloader
        train_loader = create_multiantibody_dataloader(
            train_screen, **vars(dataloader_config)
        )
        
        dataloader_config.shuffle = False
        test_loader = create_multiantibody_dataloader(
            test_screen, **vars(dataloader_config)
        )

        # Fit and predict
        model.fit(data_loader=train_loader)
        ypred = model.predict(data_loader=test_loader)
        # Attach cell line info to predictions as needed

        # Collect results
        predicted.append(ypred)  # format as needed
        fitted_model = model.model.to("cpu")
        fitted.append(copy.deepcopy(fitted_model))

    # Combine results as in leave_one_out
    out = {
        "fitted": fitted,
        "predicted": pl.DataFrame(pd.concat(predicted)),
        "importance": None 
    }
    return out

# --- Sample split training loops ---
def sample_split_mut(screen: 'ScreenBase', model: BaseModel) -> Dict:
    """ Wapper for running `sample_split` fitter by mutational background. Binary classifiers for <mutation> vs WT will be run for all ALS mutational backgrounds, excluding sporadics—which receive special treatment in sample_split.
    """
    assert screen.data is not None, "screen data not loaded"
    assert screen.metadata is not None, "screen metadata not loaded"

    mutations = screen.metadata["Mutations"].unique()
    mutations = list(set(mutations) - set(["WT", "sporadic"]))
    
    x_full = screen.data
    xmeta_full = screen.metadata
    
    out = {}
    for m in mutations:
        print(f"Training {m}...")
        select_key = [{"Mutations": ["WT", "sporadic", m]} ]
        screen = select_sample_by_feature(screen, select_key=select_key)
        
        # Set sALS cell lines as holdouts
        holdout = screen.metadata.filter(pl.col("Mutations") == "sporadic") \
            .select("CellLines") \
            .to_series() \
            .to_list()
            
        out[m] = sample_split(screen, model, holdout)
        
        screen.data = x_full
        screen.metadata = xmeta_full
        
    return out

def sample_split(
    screen: 'ScreenBase', model: BaseModel, 
    holdout: List|None=None, seed: int=47) -> Dict:
    """ Splits data into two equally sized groups. Models are fit to each group and predictions are generated for the held-out samples. Sporadics are always dropped from training and evaluated on testing.
    """
    assert screen.data is not None, "screen data not loaded"
    assert screen.metadata is not None, "screen metadata not loaded"
    
    out = {}
    
    # Define 50/50 sample split by mutation and set sporadics as hold-out
    split = cellline_split(screen, train_prop=0.5, seed=seed)

    id_holdout = screen.metadata \
        .filter(pl.col("CellLines").is_in(holdout)) \
        .select("ID") \
        .to_series()
             
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
    imp1 = model.get_importance(fit1, x)
    imp2 = model.get_importance(fit2, x)
    out['importance'] = (imp1 + imp2) / 2
        
    return out