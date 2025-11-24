from copy import copy
from typing import Dict
from typing import TYPE_CHECKING, List
from maps.multiantibody.data_loaders import create_multiantibody_dataloader
import copy
import polars as pl

if TYPE_CHECKING:
    from maps.screens import ScreenBase

def get_mutation_celllines(metadata: pl.DataFrame, mutations: List) -> List:
    """Helper function to get mutation cell lines from screen metadata."""
    cell_lines = (
        metadata
        .filter(pl.col("Mutations").is_in(mutations))
        .select("CellLines")
        .to_series()
        .to_list()
    )

    return cell_lines


def merge_metadata(screen: 'ScreenBase') -> pl.DataFrame:
    """ Merges multi-antibody metadata into a single dataframe."""
    assert screen.metadata is not None
    
    if isinstance(screen.metadata, dict):
        metadata = pl.concat([df for df in screen.metadata.values()])
    else:
        metadata = screen.metadata
    return metadata


def cellline_split(screen: 'ScreenBase', train_prop: float=0.5, type="ID") -> Dict:
    """ Splits cell lines into train / test sets by mutation."""
    assert screen.metadata is not None
    
    # Merge metadata if multi-antibody
    metadata = merge_metadata(screen)    
    
    train = (
        metadata.select(['CellLines', 'Mutations'])
        .unique()
        .sort(['Mutations', 'CellLines'])  # Add sort for deterministic ordering
        .to_pandas() 
        .groupby('Mutations')[['CellLines']]
        .apply(lambda x: 
            x.sample(frac=train_prop)) 
        .reset_index(drop=True)
    )
    
    id_train = metadata.filter(
        metadata['CellLines'].is_in(train['CellLines'])
    )
   
    id_test = metadata.filter(
        ~metadata['CellLines'].is_in(train['CellLines'])
    )
    
    # Set train / test indices relative to feature matrix
    if type == "CellLines":
        id_train = id_train.select(['CellLines']).unique().sort('CellLines')
        id_test = id_test.select(['CellLines']).unique().sort('CellLines')
        return {'id_train': id_train, 'id_test': id_test}
    else:
        id_train = id_train.select(['ID']).unique().sort('ID')
        id_test = id_test.select(['ID']).unique().sort('ID')
        return {'id_train': id_train, 'id_test': id_test}



def fit_split(screen, model, train_celllines, dataloader_config):
    """Helper function to train on one split and predict on remaining."""
    
    # Create training dataloader
    train_config = copy.deepcopy(dataloader_config)
    train_config.shuffle = True
    train_loader = create_multiantibody_dataloader(
        screen,
        select_samples=train_celllines,
        **vars(train_config)
    )
    
    # Fit model
    fitted_model = copy.deepcopy(model)
    fitted_model.fit(data_loader=train_loader)
    
    # Create prediction dataloader
    predict_config = copy.deepcopy(dataloader_config)
    predict_config.shuffle = False
    predict_config.mode = "eval"
    predict_loader = create_multiantibody_dataloader(
        screen,
        drop_samples=train_celllines,
        scalers=train_loader._get_scalers(),
        **vars(predict_config)
    )
    
    # Generate predictions
    predictions = fitted_model.predict(data_loader=predict_loader)
    scalers = train_loader._get_scalers()
    return fitted_model, predictions, scalers