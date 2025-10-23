"""
Processing modules are used for data preprocessing. All processing modules operate on and return `Screen` objects. Additional kwargs for each preprocessing function are specified through the params.json file and read by each preprocessing function as necessary.
"""
import polars as pl
from math import inf
from typing import TYPE_CHECKING
from maps.processing_utils import *

if TYPE_CHECKING:
    from maps.screens import ScreenBase


# --- Feature filters ---
def drop_na_features(x: "ScreenBase", **kwargs) -> "ScreenBase":
    """Drop columns exceeding NA threshold, replace remaining NAs with mean.
    
    Removes features with more than a specified proportion of missing values,
    then imputes remaining missing values with column means.
    
    Args:
        x (ScreenBase): Screen object with data and metadata.
        **kwargs: Additional arguments:
            - `na_prop` (float): Maximum NA proportion for columns to be kept.
                Columns with NA proportion > na_prop will be dropped.
    
    Returns:
        ScreenBase: Modified screen object with NAs handled.
    
    Example:
        ```python
        screen = drop_na_features(screen, na_prop=0.1)
        ```
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
    """Drop features with no variance (constant values).
    
    Removes columns where all values are identical, as these provide no
    information for analysis.
    
    Args:
        x (ScreenBase): Screen object with data and metadata.
    
    Returns:
        ScreenBase: Modified screen object with constant features removed.
    """    
    x.data = x.data.select([
        col for col in x.data.columns if x.data[col].n_unique() > 1
    ])
   
    return x


def select_feature_types(x: "ScreenBase", **kwargs) -> "ScreenBase":
    """Select features by regex pattern matching.
    
    Keeps only features whose names match the specified regex pattern. Useful
    for filtering to specific feature types (e.g., intensity, spot features).
    The ID column is always retained.
    
    Args:
        x (ScreenBase): Screen object with data and metadata.
        **kwargs: Additional arguments:
            - `feature_str` (str): Regex pattern to match feature names.
                Must start with '^' and end with '$'. Use '|' to match multiple
                patterns (e.g., "^.*Intensity.*$|^.*Spot.*$").
    
    Returns:
        ScreenBase: Modified screen object with only matching features.
    
    Raises:
        Exception: If regex string doesn't start with '^' and end with '$'.
    
    Example:
        ```python
        # Keep only intensity and spot features
        screen = select_feature_types(screen, 
            feature_str="^.*Intensity.*$|^.*Spot.*$")
        ```
    """
   
    feature_str = kwargs["feature_str"] 
    if not feature_str[0] == "^" and feature_str[-1] == "$":
        raise Exception("regex string must start with '^' and end with '$'")
    
    # Select features by regex matching
    feature_str = f"{feature_str}|^ID$"
    x.data = x.data.select(pl.col(feature_str))
    return x


def drop_feature_types(x: "ScreenBase", **kwargs) -> "ScreenBase":
    """Drop features by regex pattern matching.
    
    Removes features whose names match the specified regex pattern. Useful for
    excluding unwanted feature types (e.g., sum features, specific channels).
    
    Args:
        x (ScreenBase): Screen object with data and metadata.
        **kwargs: Additional arguments:
            - `feature_str` (str): Regex pattern to match feature names for removal.
                Must start with '^' and end with '$'. Use '|' to match multiple
                patterns (e.g., "^.*Sum.*$|^.*HOECHST.*$").
    
    Returns:
        ScreenBase: Modified screen object with matching features removed.
    
    Raises:
        Exception: If regex string doesn't start with '^' and end with '$'.
    
    Example:
        ```python
        # Drop sum and HOECHST features
        screen = drop_feature_types(screen, 
            feature_str="^.*Sum.*$|^.*HOECHST.*$")
        ```
    """
    
    feature_str = kwargs["feature_str"]
    if not feature_str[0] == "^" and feature_str[-1] == "$":
        raise Exception("regex string must start with '^' and end with '$'")
     
    # Drop features by regex matching
    x.data = x.data.select(pl.exclude(feature_str))
    return x


# --- Sample filters ---
def select_sample_by_feature(x: "ScreenBase", **kwargs) -> "ScreenBase":
    """Filter to samples whose metadata matches specified values.
    
    Keeps only wells/samples where metadata features match the specified values.
    Multiple criteria can be combined, and only samples matching all criteria
    are retained.
    
    Args:
        x (ScreenBase): Screen object with data and metadata.
        **kwargs: Additional arguments:
            - `select_key` (List[Dict]): List of dicts specifying selection criteria.
                Each dict has metadata column names as keys and lists of values
                to keep as values.
    
    Returns:
        ScreenBase: Modified screen object with only matching samples.
    
    Example:
        ```python
        # Keep only DMSO-treated WT and C9orf72 samples
        screen = select_sample_by_feature(screen, select_key=[
            {"Drugs": ["DMSO"]},
            {"Mutations": ["WT", "C9orf72"]}
        ])
        ```
    """
    
    for item in kwargs["select_key"]:
        for k, v in item.items():
            x.metadata = x.metadata.filter(pl.col(k).is_in(v))
            x.data = x.data.filter(pl.col('ID').is_in(x.metadata['ID']))
            
    return x


def drop_sample_by_feature(x: "ScreenBase", **kwargs) -> "ScreenBase":
    """Filter out samples whose metadata matches specified values.
    
    Removes wells/samples where metadata features match the specified values.
    Multiple criteria can be combined to exclude various sample types.
    
    Args:
        x (ScreenBase): Screen object with data and metadata.
        **kwargs: Additional arguments:
            - `drop_key` (List[Dict]): List of dicts specifying exclusion criteria.
                Each dict has metadata column names as keys and lists of values
                to exclude as values.
    
    Returns:
        ScreenBase: Modified screen object with matching samples removed.
    
    Example:
        ```python
        # Drop specific problematic cell lines
        screen = drop_sample_by_feature(screen, drop_key=[
            {"CellLines": ["C9014", "NS048", "FTD37"]},
            {"Mutations": ["TDP43"]}
        ])
        ```
    """
    
    for item in kwargs["drop_key"]:
        for k, v in item.items():
            x.metadata = x.metadata.filter(~pl.col(k).is_in(v))
            x.data = x.data.filter(pl.col('ID').is_in(x.metadata['ID']))
            
    return x


def drop_cells_by_count(x: "ScreenBase", **kwargs) -> "ScreenBase":
    """Drop cell lines based on average cell count thresholds.
    
    Filters out cell lines with average cell counts outside the specified range.
    Useful for quality control to remove wells with poor imaging or cell health.
    
    Args:
        x (ScreenBase): Screen object with data and metadata.
        **kwargs: Additional arguments:
            - `min_count` (int, optional): Minimum average cell count. Defaults to 0.
            - `max_count` (int, optional): Maximum average cell count. Defaults to inf.
    
    Returns:
        ScreenBase: Modified screen object with filtered cell lines.
    
    Example:
        ```python
        # Keep only cell lines with 100-1000 cells on average
        screen = drop_cells_by_count(screen, min_count=100, max_count=1000)
        ```
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
    """Drop cells by feature quantile filtering within ID groups.
    
    Removes individual cells with extreme feature values (outliers) based on
    quantile thresholds. Filtering is applied per well (ID group) to handle
    well-to-well variation. Cells in the lower and upper quantiles are removed.
    
    Args:
        x (ScreenBase): Screen object with data and metadata.
        **kwargs: Additional arguments:
            - `feature_filters` (Dict[str, float]): Dict mapping feature names to
                quantile thresholds. Cells below quantile and above (1-quantile)
                are removed for each feature.
    
    Returns:
        ScreenBase: Modified screen object with outlier cells removed.
    
    Example:
        ```python
        # Remove cells in bottom/top 5% for area features
        screen = drop_cells_by_feature_qt(screen, feature_filters={
            "Nucleus_Region_Area_[µm²]": 0.05,
            "Cell_Region_Area_[µm²]": 0.05
        })
        ```
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
    """Randomly subsample n cells from each well (ID group).
    
    Reduces the number of cells per well to a fixed number, useful for
    computational efficiency or balanced sampling across wells.
    
    Args:
        x (ScreenBase): Screen object with data and metadata.
        **kwargs: Additional arguments:
            - `n` (int, optional): Number of cells to sample per well. Defaults to 100.
            - `seed` (int, optional): Random seed for reproducibility. Defaults to 47.
    
    Returns:
        ScreenBase: Modified screen object with subsampled cells.
    
    Example:
        ```python
        # Sample 250 cells per well
        screen = subsample_rows_by_id(screen, n=250, seed=42)
        ```
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


# --- Feature transforms ---
def pca_transform(x: "ScreenBase", **kwargs) -> "ScreenBase":
    """Perform PCA transformation on features grouped by marker.
    
    Applies PCA within each feature group (e.g., by marker/antibody) and retains
    principal components explaining the specified variance threshold. Reduces
    dimensionality while preserving group structure.
    
    Args:
        x (ScreenBase): Screen object with data and metadata.
        **kwargs: Additional arguments:
            - `alpha` (float): Minimum percent variance to explain (0-1).
            - `groups` (List[str]): List of regex patterns for grouping features.
                Features matching each pattern form a group for separate PCA.
                Unmatched features are grouped separately.
    
    Returns:
        ScreenBase: Modified screen object with PCA-transformed features.
    
    Example:
        ```python
        # PCA by marker, keeping 90% variance
        screen = pca_transform(screen, 
            alpha=0.9, 
            groups=["^.*FUS.*$", "^.*EEA1.*$"])
        ```
    """
    alpha = kwargs["alpha"]
    groups = kwargs["groups"]
    x.data = pca_by_group(x.data, groups, alpha)
    return x

def group_markers(x: "ScreenBase", **kwargs) -> "ScreenBase":
    """Reformat data by grouping features into specified marker groups.
    
    Organizes features into groups based on regex matching. Useful for
    structuring multi-marker data or organizing features by type.
    
    Args:
        x (ScreenBase): Screen object with data and metadata.
        **kwargs: Additional arguments:
            - `groups` (List[str]): List of regex patterns for grouping features.
                Features matching each pattern are grouped together. Unmatched
                features are grouped separately.
    
    Returns:
        ScreenBase: Modified screen object with grouped features.
    
    Example:
        ```python
        # Group features by marker
        screen = group_markers(screen, 
            groups=["^.*FUS.*$", "^.*EEA1.*$", "^.*COX.*$"])
        ```
    """
    groups = kwargs["groups"]
    x.data = group_features_by_marker(x.data, groups)
    return x

def standardize_features(x: "ScreenBase") -> "ScreenBase":
    """Standardize features by removing mean and scaling to unit variance.
    
    Z-score normalization: subtracts mean and divides by standard deviation
    for each feature. The ID column is preserved without transformation.
    
    Args:
        x (ScreenBase): Screen object with data and metadata.
    
    Returns:
        ScreenBase: Modified screen object with standardized features.
    
    Example:
        ```python
        screen = standardize_features(screen)
        ```
    """
    
    numeric_cols = [c for c in x.data.columns if c != "ID"]
    x.data = x.data.with_columns([
        ((pl.col(c) - pl.col(c).mean()) / (pl.col(c).std() + 1e-6)).alias(c)
        for c in numeric_cols
    ])
    
    return x       