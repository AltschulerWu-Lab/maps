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

import torch
from typing import Dict, List
from torch.utils.data import DataLoader
import pandas as pd

def integrated_gradients(
    dataloader: DataLoader,
    target_class,
    fit: torch.nn.Module,
    n_steps: int = 50,
    baseline: str = 'zeros'
):
    """ Compute integrated gradients from fitted model
    
    Returns:
        pd.DataFrame: DataFrame with columns ['CellLine', 'Class', 'Feature_0', 'Feature_1', ...]
                     Each row represents one cell line-class combination with averaged IG values
    """
    fit.eval()
    device = next(fit.parameters()).device
    integrated_grads = {}
    cell_lines = []
     
    for batch in dataloader:
        # Prepare batch data - dictionary of antibody tensors
        x_dict = {ab: batch[ab][0].to(device) for ab in batch}

        ab = list(batch.keys())[0]
        cell_lines.append(batch[ab][-1])

        # Initialize integrated gradients dict
        for ab in x_dict:
            if ab not in integrated_grads:
                integrated_grads[ab] = []

        # Create baseline dict
        if baseline == 'zeros':
            baseline_dict = {ab: torch.zeros_like(x_dict[ab]) for ab in x_dict}
        elif baseline == 'random':
            baseline_dict = {ab: torch.rand_like(x_dict[ab]) * 0.1 for ab in x_dict}
        else:
            raise ValueError(f"Unknown baseline type: {baseline}")

        # Get batch size from first antibody
        batch_size = x_dict[list(x_dict.keys())[0]].shape[0]

        # Determine if target_class is a list/array of classes
        if isinstance(target_class, (list, tuple, np.ndarray)):
            target_classes = list(target_class)
        else:
            target_classes = [target_class]

        # Process each sample in batch
        for i in range(batch_size):
            # Extract single sample for each antibody
            x_sample = {ab: x_dict[ab][i:i+1] for ab in x_dict}
            baseline_sample = {ab: baseline_dict[ab][i:i+1] for ab in x_dict}

            # For each class, compute integrated gradients and stack along last axis
            class_grads = {ab: [] for ab in x_sample}
            for class_idx in target_classes:
                # Initialize integrated gradients for this sample/class
                integrated_grad_sample = {ab: torch.zeros_like(x_sample[ab]) for ab in x_sample}

                # Compute integrated gradients
                for step in range(n_steps + 1):
                    # Interpolate between baseline and input for all antibodies
                    alpha = step / n_steps
                    x_interp = {}
                    for ab in x_sample:
                        x_interp[ab] = baseline_sample[ab] + alpha * (x_sample[ab] - baseline_sample[ab])
                        x_interp[ab].requires_grad_(True)

                    # Forward pass through model
                    _, line_logits = fit(x_interp)

                    # Get target class score
                    target_score = line_logits[0, class_idx]

                    # Compute gradients
                    fit.zero_grad()
                    target_score.backward()

                    # Accumulate gradients for each antibody
                    if step > 0:  # Skip baseline (step 0)
                        for ab in x_interp:
                            integrated_grad_sample[ab] += x_interp[ab].grad.data

                # Average and scale by input difference for each antibody
                for ab in x_sample:
                    integrated_grad = ((x_sample[ab] - baseline_sample[ab]) *
                                      integrated_grad_sample[ab] / n_steps)
                    # Store as numpy array for this class
                    class_grads[ab].append(integrated_grad.cpu().detach().numpy().squeeze())

            # Stack class gradients along last axis (-1)
            for ab in x_sample:
                stacked = np.stack(class_grads[ab], axis=-1)  # shape (..., num_classes)
                integrated_grads[ab].append(stacked)

    # Convert to DataFrame format
    results = []
    
    # Flatten cell_lines list (in case some are lists themselves)
    flat_cell_lines = []
    for cl in cell_lines:
        if isinstance(cl, (list, tuple)):
            flat_cell_lines.extend(cl)
        else:
            flat_cell_lines.append(cl)
    
    # Process each cell line
    for cell_line_idx, cell_line in enumerate(flat_cell_lines):
        # For each class
        for class_idx in range(len(target_classes)):
            # Concatenate features from all antibodies for this cell line and class
            all_features = []
            for ab in sorted(integrated_grads.keys()):
                # Shape: (n_cells, n_features, n_classes)
                ig_array = integrated_grads[ab][cell_line_idx]
                
                # Average over cells and extract this class
                # ig_array[..., class_idx] gives (n_cells, n_features)
                # Mean over cells (axis 0) gives (n_features,)
                features_for_class = ig_array[..., class_idx].mean(axis=0)
                all_features.append(features_for_class)
            
            # Concatenate all antibody features
            all_features = np.concatenate(all_features)
            
            # Create row
            row_dict = {'CellLine': cell_line, 'Class': target_classes[class_idx]}
            for feat_idx, feat_val in enumerate(all_features):
                row_dict[f'Feature_{feat_idx}'] = feat_val
            
            results.append(row_dict)
    
    return pd.DataFrame(results)
