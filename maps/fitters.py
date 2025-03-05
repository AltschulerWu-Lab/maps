""" Fitter functions are used to run classification pipelines used for MAP-scoring analysis. Fitters define data splits, how model(s) should be trained relative to these sample splits, how predictions should be generated relative to these splits.
"""
from typing import Dict
from maps.models import BaseModel
from maps.processing import select_sample_by_feature
import polars as pl
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maps.screens import ScreenBase

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
        out[m] = sample_split(screen, model)
        
        screen.data = x_full
        screen.metadata = xmeta_full
        
    return out
        
    
def sample_split(screen: 'ScreenBase', model: BaseModel) -> Dict:
    """ Splits data into two equally sized groups. Models are fit to each group and predictions are generated for the held-out samples. Sporadics are always dropped from training and evaluated on testing.
    """
    
    out = {}
    
    # Define 50/50 sample split by mutation and set sporadics as hold-out
    split = cellline_split(screen)

    sporadics = screen.metadata \
        .filter(pl.col("Mutations") \
        .is_in(["sporadic"]))
        
    id_sporadic = sporadics["ID"]
    
    
    # Initialize data for model fitting
    y = screen.get_response() 
    x = screen.get_data()
   
    # Fit models to each sample split
    id1 = split["id_train"].filter(~split["id_train"].is_in(id_sporadic))
    fit1 = model.fit(x=x, y=y, id_train=id1)
    
    id2 = split["id_test"].filter(~split["id_test"].is_in(id_sporadic))
    fit2 = model.fit(x=x, y=y, id_train=id2)

    out['fitted'] = [fit1, fit2]
    
    # Generate model predictions
    id2 = pl.concat([split["id_test"], id_sporadic]).unique()
    yp2 = model.predict(fit1, x, id_test=id2)  
    yp2 = yp2.with_columns((pl.col("Ypred") - pl.col("Ypred").mean() +  0.5))
    
    id1 = pl.concat([split["id_train"], id_sporadic]).unique() 
    yp1 = model.predict(fit2, x, id_test=split['id_train'])
    yp1 = yp1.with_columns((pl.col("Ypred") - pl.col("Ypred").mean() +  0.5))

    out['predicted'] = pl.concat([yp1, yp2]).join(screen.metadata, on="ID")
    
    # Generate model feature importance
    imp1 = model.get_importance(fit1, x)
    imp2 = model.get_importance(fit2, x)
    out['importance'] = (imp1 + imp2) / 2
        
    return out


def cellline_split(screen: 'ScreenBase', train_prop: float=0.5) -> Dict:
    """ Splits cell lines into train / test sets by mutation."""
        
    cols = ['CellLines', 'Mutations']
    key = screen.metadata.select(cols).unique()
    
    train = key.to_pandas() \
        .groupby('Mutations') \
        .apply(lambda x: x.sample(frac=train_prop), include_groups=False) \
        .reset_index(drop=True)

    id_train = screen.metadata.filter(
        screen.metadata['CellLines'].is_in(train['CellLines'])
    )
   
    id_test = screen.metadata.filter(
        ~screen.metadata['CellLines'].is_in(train['CellLines'])
    )
    
    # Set train / test indices relative to feature matrix
    return {'id_train': id_train["ID"], 'id_test': id_test["ID"]}