import re
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, List

def qt_filter_feature(df: pl.DataFrame, feature_filters: Dict):
    """Filters df at alpha / 1 - alpha quantile of selected features"""
    group_lst = []

    for f, alpha in feature_filters.items():
        q_low = df[f].quantile(alpha)
        q_high = df[f].quantile(1-alpha)
        group_lst.append((df[f] >= q_low) & (df[f] <= q_high))
    
    return df.filter(pl.all_horizontal(group_lst))


def pca(df: pl.DataFrame, alpha: float=0.95):
    """Runs PCA and maintains features explaining alpha % variance."""
    # Drop non-numeric column
    ids = df["ID"]
    df_numeric = df.drop("ID")
    
    # Scale the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric.to_pandas())
    
    # Perform PCA
    pca = PCA(n_components=alpha)
    principal_components = pca.fit_transform(df_scaled)
    
    # Create a DataFrame with the principal components
    pc_df = pl.DataFrame(principal_components)
    pc_df = pc_df.with_columns(ids)
    
    # Rename columns from 'column_X' to 'PC_X'
    pc_df = pc_df.rename({
        f"column_{i}": f"PC{i}" for i in range(pc_df.shape[1]-1)
    })
    
    return pc_df


def pca_by_group(df: pl.DataFrame, groups: List, alpha: float=0.95):
    """Runs PCA by column groups and maintains features explaining alpha % 
    variance within each group."""
    result = []
    remaining_columns = set(df.columns) - set(["ID"])
    
    for group in groups:
        # Select columns matching the group by regex
        group_columns = [col for col in df.columns if re.search(group, col)]
        remaining_columns -= set(group_columns)
        
        if group_columns:
            group_df = df.select(group_columns + ["ID"])
            pc_df = pca(group_df, alpha).drop("ID")
            pc_df = pc_df.rename({
                col:f"{group}_{col}" for col in pc_df.columns
            })
            result.append(pc_df)
    
    # Handle columns that match no entry of groups
    if remaining_columns:
        remaining_df = df.select(list(remaining_columns) + ["ID"])
        pc_df = pca(remaining_df, alpha)
        result.append(pc_df)
    
    return pl.concat(result, how="horizontal")

