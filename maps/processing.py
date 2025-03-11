"""
Processing modules are used for data preprocessing. All processing modules operate on and return Screen class objects. Additional kwargs for each preprocessing function are specified through the params.json file and read by each preprocessing function as necessary.
"""
import polars as pl
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maps.screens import ScreenBase
    
def drop_na_features(x: "ScreenBase", **kwargs) -> "ScreenBase":
    "Drop columns exceeding NA threshold, replace with mean otherwise"
    
    df = x.data 
    na_prop = kwargs['na_prop']    
    
    # Drop columns with NA proportion greater than threshold
    na_prop_cols = df.select(pl.all().is_null().sum() / len(df)).to_dict()
    cols_drop = [col for col, prop in na_prop_cols.items() if prop[0] > na_prop]
    df = df.drop(cols_drop)
   
    # Replace remaining NA values with mean 
    x.data = df.fill_null(strategy="mean")
        
    return x

def drop_constant_features(x: "ScreenBase") -> "ScreenBase":
    " Drop constant features"
    
    df = x.data
    x.data = df.select([
        col for col in df.columns if df[col].n_unique() > 1
    ])
   
    return x

def select_feature_types(x: "ScreenBase", **kwargs) -> "ScreenBase":
    "Select features by regex match"
   
    feature_str = kwargs["feature_str"] 
    if not feature_str[0] == "^" and feature_str[-1] == "$":
        raise Exception("regex string must start with '^' and end with '$'")
    
    # Select features by regex matching
    feature_str = f"{feature_str}|^ID$"
    x.data = x.data.select(pl.col(feature_str))
    return x

def drop_feature_types(x: "ScreenBase", **kwargs) -> "ScreenBase":
    "Drop features by regex match"
    
    feature_str = kwargs["feature_str"]
    if not feature_str[0] == "^" and feature_str[-1] == "$":
        raise Exception("regex string must start with '^' and end with '$'")
     
    # Drop features by regex matching
    x.data = x.data.select(pl.exclude(feature_str))
    return x

def select_sample_by_feature(x: "ScreenBase", **kwargs) -> "ScreenBase":
    "Filter to samples whose metadata feature matches specified value"
    
    for item in kwargs["select_key"]:
        for k, v in item.items():
            x.metadata = x.metadata.filter(pl.col(k).is_in(v))
            x.data = x.data.filter(pl.col('ID').is_in(x.metadata['ID']))
            
    return x

def drop_sample_by_feature(x: "ScreenBase", **kwargs) -> "ScreenBase":
    "Filter to samples whose metadata feature does not match specified value"
    
    for item in kwargs["drop_key"]:
        for k, v in item.items():
            x.metadata = x.metadata.filter(~pl.col(k).is_in(v))
            x.data = x.data.filter(pl.col('ID').is_in(x.metadata['ID']))
            
    return x