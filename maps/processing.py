"""
Processing modules are used for data preprocessing. All processing modules operate on and return `Screen` objects. Additional kwargs for each preprocessing function are specified through the params.json file and read by each preprocessing function as necessary.
"""
import polars as pl
from math import inf
from typing import TYPE_CHECKING
from maps.processing_utils import *

if TYPE_CHECKING:
    from maps.screens import ScreenBase


# Feature filters
def drop_na_features(x: "ScreenBase", **kwargs) -> "ScreenBase":
    """Drop columns exceeding NA threshold, replace with mean otherwise. 
    
    Additional kwargs:
    
        na_prop (float): max na proportion for columns kept in dataset.
    """
    
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
    """Drops constant features from screen data."""    
    x.data = x.data.select([
        col for col in x.data.columns if x.data[col].n_unique() > 1
    ])
   
    return x


def select_feature_types(x: "ScreenBase", **kwargs) -> "ScreenBase":
    """Select features by regex match. 
    
    Additional kwargs:
    
        feature_str (str): regex string to match features on. To select
        multiple features, separate terms with `|`. Each term must start with 
        `^` and end with `$`
    """
   
    feature_str = kwargs["feature_str"] 
    if not feature_str[0] == "^" and feature_str[-1] == "$":
        raise Exception("regex string must start with '^' and end with '$'")
    
    # Select features by regex matching
    feature_str = f"{feature_str}|^ID$"
    x.data = x.data.select(pl.col(feature_str))
    return x


def drop_feature_types(x: "ScreenBase", **kwargs) -> "ScreenBase":
    """Drop features by regex match. 
    
    Additional kwargs:
    
        feature_str (str): regex string to drop features on. To select multiple 
        features, separate terms with `|`. Each term must start with `^` and 
        end with `$`
    """
    
    feature_str = kwargs["feature_str"]
    if not feature_str[0] == "^" and feature_str[-1] == "$":
        raise Exception("regex string must start with '^' and end with '$'")
     
    # Drop features by regex matching
    x.data = x.data.select(pl.exclude(feature_str))
    return x


# Sample filters
def select_sample_by_feature(x: "ScreenBase", **kwargs) -> "ScreenBase":
    """Filter to samples whose metadata feature matches specified value. 
    
    Additional kwargs:
    
        select_key (list): each list item being a dict, keys specifying the     
        metadata feature and values the list of feature values to be selected.
    """
    
    for item in kwargs["select_key"]:
        for k, v in item.items():
            x.metadata = x.metadata.filter(pl.col(k).is_in(v))
            x.data = x.data.filter(pl.col('ID').is_in(x.metadata['ID']))
            
    return x


def drop_sample_by_feature(x: "ScreenBase", **kwargs) -> "ScreenBase":
    """Filter to samples whose metadata feature does not match specified value. 
    
    Additional kwargs:
    
        drop_key (list): each list item being a dict, keys specifying the     
        metadata feature and values the list of feature values to be dropped.
    """
    
    for item in kwargs["drop_key"]:
        for k, v in item.items():
            x.metadata = x.metadata.filter(~pl.col(k).is_in(v))
            x.data = x.data.filter(pl.col('ID').is_in(x.metadata['ID']))
            
    return x


def drop_cells_by_count(x: "ScreenBase", **kwargs) -> "ScreenBase":
    """Drop cells by applying feature quantile filtering (at quantile and 1 - quantile levels). Feature quantiles are computed by ID group. Additional kwargs:
    
        min_count (int): minimum average cell count for cell line to be kept.
        
        max_count (int): maximum average cell count for cell line to be kept.
    """
    min_count = kwargs.get("min_count", 0) 
    max_count = kwargs.get("max_count", inf)
    
    meta_cell = x.metadata \
        .group_by("CellLines") \
        .agg(pl.col("NCells") \
        .mean() \
        .alias("NCells"))
        
    meta_cell = meta_cell.filter(
        (pl.col("NCells") >= min_count) & (pl.col("NCells") <= max_count)
    )
    
    x.metadata = x.metadata \
        .filter(pl.col("CellLines").is_in(meta_cell["CellLines"]))
    
    x.data = x.data \
        .filter(pl.col("ID").is_in(x.metadata["ID"]))
    
    return x


def drop_cells_by_feature_qt(x: "ScreenBase", **kwargs) -> "ScreenBase":
    """Drop cells by applying feature quantile filtering (at quantile and 1 - 
    quantile levels). Feature quantiles are computed by ID group. 
    
    Additional kwargs:
    
        feature_filters (dict): key, value pairs giving feature to apply 
        thresholding to and quantile level.
    """
    feature_filters = kwargs.get("feature_filters") 
    df = x.data
    
    # Group by ID and apply filtering to each group
    filtered_dfs = [
        qt_filter_feature(dfg[1], feature_filters) for dfg in df.group_by('ID')
    ]
    
    x.data = pl.concat(filtered_dfs)
    return x


def subsample_rows_by_id(x: "ScreenBase", **kwargs) -> "ScreenBase":
    """Sample n rows from each ID group in a polars DataFrame. 
    
    Additional kwargs:
    
        n (int): number of rows sampled per ID
        
        seed (int): random sampling seed.
    """
    n = kwargs.get("n", 100)
    seed = kwargs.get("seed", 47)

    df = x.data.to_pandas() \
        .groupby("ID") \
        .apply(lambda group: group.sample(
            n=min(n, len(group)), 
            random_state=seed
        ), include_groups=True)
    
    x.data = pl.DataFrame(df)
    return x


# Feature transforms
def pca_transform(x: "ScreenBase", **kwargs) -> "ScreenBase":
    """Performs PCA transformation on features by group (e.g., marker) and 
    maintains PCs that explain specified % variance within each group. 
    
    Additional kwargs:
    
        alpha (float): percent variance threshold
    
        groups (list): list of column regex matches for grouping data. Any
        column matching no entries of groups will be grouped.
    """
    alpha = kwargs["alpha"]
    groups = kwargs["groups"]
    x.data = pca_by_group(x.data, groups, alpha)
    return x


