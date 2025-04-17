import polars as pl
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def balanced_sample(
    x: pl.DataFrame, y: np.ndarray, type: str="down", random_state: int=47): 
    """Take a balanced sample of binary data through up or down sampling. 
    
    Args:
    
        x (pl.DataFrame): DataFrame containing the data to be sampled.
        y (np.ndarray): Array of labels corresponding to the data.
        type (str): Type of sampling to perform. "up" for oversampling, "down" for undersampling.
        random_state (int): Random state for reproducibility.
    
    """
    
    if type == "up":
        sampler = RandomOverSampler(random_state=random_state)
    elif type == "down":
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError("type must be 'up' or 'down'")
    
    idx_array = np.arange(len(x))
    sampled_idx, sampled_y = sampler.fit_resample(idx_array.reshape(-1, 1), y)
    sampled_x = x[sampled_idx.ravel()]

    return sampled_x, sampled_y