from typing import Dict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maps.screens import ScreenBase
    
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