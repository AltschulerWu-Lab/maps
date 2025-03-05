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


if False: 
    # PCA Transform
    def pca_transform(x: ImageScreen, **kwargs):
        #(xlist, groups=None, pct_var_thr=0.95):
        
        xfeat = x.data.select(pl.exclude('ID'))
        feature_meta = make_feature_meta(xfeat)

        if kwargs['groups']:
            # Group features by the specified columns
            groups = kwargs['groups']
            group_cols = feature_meta.select(groups).to_pandas()
            unique_groups = group_cols.drop_duplicates()
            
            xg = []
            for _, group in unique_groups.iterrows():
                group_features = feature_meta.filter(pl.col(groups).is_in(group.tolist()))['Feature']
                xfeati = xfeat.select(group_features)
                
                # Standard scaling
                pca = PCA()
                xfeati = StandardScaler().fit_transform(xfeati.to_pandas())
                pca_result = pca.fit_transform(xfeati)
                
                pct_var = np.cumsum(pca.explained_variance_ratio_)
                id = np.where(pct_var > kwargs['pct_var_thr'])[0][0]
                xg.append(pca_result[:, :id])
            
            x.data = pl.DataFrame(np.hstack(xg), schema={'ID': x.data['ID'].to_list()})
        return x


    # Drop Sample by CellCount
    def drop_sample_by_cellcount(xlist, min_cell_count=100):
        assert 'x' in xlist
        assert 'xmeta' in xlist
        
        cell_lines_drop = xlist['xmeta'].filter(pl.col('Drugs') == 'DMSO') \
            .groupby('CellLines').agg(pl.col('N').mean())
        cell_lines_drop = cell_lines_drop.filter(pl.col('N') < min_cell_count)['CellLines']
        
        id_drop = xlist['xmeta'].filter(pl.col('CellLines').is_in(cell_lines_drop))['ID']
        
        xlist['xmeta'] = xlist['xmeta'].filter(~pl.col('ID').is_in(id_drop))
        xlist['x'] = xlist['x'].filter(~pl.col('ID').is_in(id_drop))
        return xlist


    # Make Feature Meta
    def make_feature_meta(x):
        # NOTE: these are mostly incorrect, but probably not needed
        if 'ID' in x.columns:
            x = x.select(pl.exclude('ID'))
        
        cols = x.columns
        regions = [col.split('_Region')[0] for col in cols]
        markers = [col.split('Alexa_') for col in cols]
        markers = [m[1].split('_')[0] if len(m) > 1 else None for m in markers]

        Type = None
        stats = [col.split('_')[-1] for col in cols]

        feature_meta = pl.DataFrame({
            'Feature': cols,
            'Region': regions,
            'Marker': markers,
            'Type': types,
            'Stat': stats
        })
        
        return feature_meta
