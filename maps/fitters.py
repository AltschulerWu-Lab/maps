""" Fitter functions are used to run classification pipelines used for MAP-scoring analysis. Fitters define data splits, how model(s) should be trained relative to these sample splits, how predictions should be generated relative to these splits.
"""

import polars as pl
import pandas as pd

from typing import Dict
from maps.models import BaseModel
from maps.processing import select_sample_by_feature
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from maps.screens import ScreenBase

def leave_one_out_mut(screen: 'ScreenBase', model: BaseModel) -> Dict:
    """ Wapper for running leave_one_out fitter by mutational background. Binary classifiers for <mutation> vs WT will be run for all ALS mutational backgrounds, excluding sporadics—which receive special treatment in sample_split.
    """
    
    mutations = screen.metadata["Mutations"].unique()
    mutations = list(set(mutations) - set(["WT", "sporadic"]))
    out = {}
    
    for m in mutations:
        print(f"Training {m}...")
        select_key = [{"Mutations": ["WT", "sporadic", m]} ]
        screen_m = select_sample_by_feature(screen, select_key=select_key)
        
        # Set sALS cell lines as holdouts
        holdout = screen_m.metadata.filter(pl.col("Mutations") == "sporadic") \
            .select("CellLines") \
            .to_series() \
            .to_list()
            
        out[m] = leave_one_out(screen_m, model, holdout)
        
        
    return out

def leave_one_out(
    screen: "ScreenBase", model: BaseModel, holdout: List|None=None):
    """Fit models with one cell line removed and evaluates prediction on 
    hold-out cell line. Optionally include additional set of cell lines as holdouts.
    """

    y = screen.get_response() 
    x = screen.get_data()
    
    # Get unique cell lines and exclude test cell lines
    cell_lines = screen.metadata \
        .select("CellLines") \
        .unique() \
        .to_series() \
        .to_list()
    
    if holdout is not None:
        cell_lines = [cl for cl in cell_lines if cl not in holdout]
    
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
            .select("ID")
            
        id_test = screen.metadata \
            .filter(pl.col("CellLines").is_in(cell_lines_test)) \
            .select("ID")
        
        # Fit model and make predictions
        fitted_cl = model.fit(x=x, y=y, id_train=id_train)
        predicted_cl = model.predict(fitted_cl, x=x, id_test=id_test)
        predicted_cl= predicted_cl.with_columns(pl.lit(cl).alias("Holdout"))
        
        # Merge results
        predicted.append(predicted_cl.join(screen.metadata, on="ID"))
        fitted.append(fitted_cl)
        importance.append(model.get_importance(fitted_cl, x))
      
    out = {}  
    out["fitted"] = fitted
    out["predicted"] = pl.concat(predicted)
    out["importance"] = pd.concat([pd.Series(p) for p in importance], axis=1)
    return out

    
def sample_split_mut(screen: 'ScreenBase', model: BaseModel) -> Dict:
    """ Wapper for running sample_split fitter by mutational background. Binary classifiers for <mutation> vs WT will be run for all ALS mutational backgrounds, excluding sporadics—which receive special treatment in sample_split.
    """
    
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
    
    out = {}
    
    # Define 50/50 sample split by mutation and set sporadics as hold-out
    split = cellline_split(screen, seed)

    id_holdout = screen.metadata \
        .filter(pl.col("CellLines").is_in(holdout)) \
        .select("ID") \
        .to_series() \
        .to_list()
             
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


def cellline_split(
    screen: 'ScreenBase', train_prop: float=0.5, seed: int=47) -> Dict:
    """ Splits cell lines into train / test sets by mutation."""
        
    cols = ['CellLines', 'Mutations']
    key = screen.metadata.select(cols).unique()
    
    train = key.to_pandas() \
        .groupby('Mutations') \
        .apply(lambda x: 
            x.sample(
                frac=train_prop, 
                random_state=seed), 
            include_groups=False) \
        .reset_index(drop=True)

    id_train = screen.metadata.filter(
        screen.metadata['CellLines'].is_in(train['CellLines'])
    )
   
    id_test = screen.metadata.filter(
        ~screen.metadata['CellLines'].is_in(train['CellLines'])
    )
    
    # Set train / test indices relative to feature matrix
    return {'id_train': id_train["ID"], 'id_test': id_test["ID"]}