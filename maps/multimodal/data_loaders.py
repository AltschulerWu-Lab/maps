"""
PyTorch DataLoader for ImageScreenMultimodal data.

This module provides a PyTorch Dataset and DataLoader for handling multimodal 
imaging data from ImageScreenMultimodal objects. The DataLoader returns 
separate tensors for each antibody as dictionaries rather than combining 
all antibodies into a single tensor.
"""

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List
import random
from maps.screens import ImageScreenMultimodal


class ImagingDataset(Dataset):
    """
    PyTorch Dataset for imaging data from one "modality" (e.g., marker).
    
    Creates a dataset that samples data according to response and grouping variables. Handles sampling of single cell-level features and responses by grouping variable (e.g., cell lines).
    """
    
    def __init__(
        self, 
        data: pl.DataFrame, 
        metadata: pl.DataFrame,
        antibody: str,
        response: str = "Mutations",
        grouping: str = "CellLines", 
        response_map: Optional[Dict[str, int]] = None,
        n_cells: int = 10,
        seed: Optional[int] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data: Polars DataFrame containing imaging data for the antibody
            metadata: Polars DataFrame containing metadata for the antibody
            antibody: Antibody to use for the dataset
            response: Column name for response variable (default: "Mutations")
            grouping: Column name for grouping variable (default: "cell_lines")
            response_map: Mapping from response strings to numeric labels
            n_cells: Number of samples to sample per group
            seed: Random seed for reproducible sampling
        """
        self.data = data
        self.metadata = metadata
        self.response = response
        self.antibody = antibody
        self.grouping = grouping
        self.n_cells = n_cells
        self.response_map = response_map  
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Prepare the dataset
        self._prepare_data()
        
    def _prepare_data(self):
        
        # Map response strings to numeric labels
        if self.response_map is None:
            response_values = self.metadata[self.response].unique()
            self.response_map = {
                resp: i for i, resp in enumerate(response_values)
            }
        
        self.metadata = self.metadata.with_columns(
            pl.col(self.response).replace(self.response_map).alias("Label")
        )
        
        self.metadata = self.metadata.with_columns(
            pl.col("Label").cast(pl.Int64)
        )
    
    def _get_features(self, idx: int):
        """
        Get features for a specific antibody at a given index.
        """
        id_select = self.metadata["ID"].to_list()[idx]
        x = self.data.filter(pl.col("ID") == id_select)

        # Sample self.n_cells entries from x (with replacement if needed)
        n_rows = x.height
        if n_rows >= self.n_cells:
            idxs = np.random.choice(n_rows, self.n_cells, replace=False)
        else:
            idxs = np.random.choice(n_rows, self.n_cells, replace=True)
        
        x_sampled = pl.concat([x.slice(int(i), 1) for i in idxs]).drop("ID")
        feature_groups = [col.split("_")[0] for col in x_sampled.columns]
        x_sampled = torch.tensor(x_sampled.to_numpy(), dtype=torch.float32) 
        return x_sampled, feature_groups

    def _get_response(self, idx: int):
        """
        Get response label for a specific antibody at a given index.
        """
        y = self.metadata["Label"].to_numpy()[idx]
        return torch.tensor(y, dtype=torch.long)
        # if self.betas is None:
        #     self.betas = torch.randn(x.shape[1], dtype=torch.float32)
        # 
        # # Use mean of features across cells for deterministic response
        # x_mean = x.mean(dim=0)  # Average across cells
        # p = torch.sigmoid(x_mean @ self.betas)
        # y = (p > 0.5).long()  # Deterministic threshold instead of random sampling
        # return y
    
        
    def __len__(self):
        """Return the length of the dataset."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Get a single item from the dataset."""
        x, feat_group = self._get_features(idx)
        y = self._get_response(idx)

        return x, y, feat_group


class MultiAntibodyLoader(DataLoader):
    """
    Wraps multiple antibody-specific datasets in a single data loader. Batches are generated as dicts keyed by antibody. Data are generated for cell lines shared across antibodies.
    """
    def __init__(self, datset_dict, shuffle: bool = True):
        self.datset_dict = datset_dict
        self.keys = list(datset_dict.keys())
        
        # Find cell lines shared across antibodies
        cell_lines = {}
        for ab in self.keys:
            ds = datset_dict[ab].dataset
            cl_mut_pairs = set(zip(
                ds.metadata['CellLines'].to_list(), 
                ds.metadata['Mutations'].to_list()
            ))
            cell_lines[ab] = cl_mut_pairs
       

        common_cell_lines = sorted(
            set.intersection(*cell_lines.values())
        )
        
        # Count mutations representation for balancing
        self.cell_lines = pl.DataFrame(
            {"CellLines": [cl[0] for cl in common_cell_lines],
             "Mutations": [cl[1] for cl in common_cell_lines]}
        )
       
        self.mutation_counts = self.cell_lines.group_by("Mutations").agg(
            [pl.count("CellLines").alias("count")]
        )
        
        
        self.balanced_cell_lines = self._balance_sample() 
        self.batch_size = datset_dict[self.keys[0]].batch_size
        self.num_batches = len(self.balanced_cell_lines) // self.batch_size
        self._batch_idx = 0
        self.shuffle = shuffle
        self.mode = "train"

    def __iter__(self):
        self._batch_idx = 0
        self.balanced_cell_lines = self._balance_sample() 
        
        # Shuffle cell lines at the start of each epoch if enabled
        if self.shuffle:
            self.balanced_cell_lines = self.balanced_cell_lines.sample(
                fraction=1.0, shuffle=True
            )   
        
        return self
    
    def __len__(self):
        return self.num_batches
    
    def __next__(self):
        if self.mode == "train":
            return self.__next_train__()
        elif self.mode == "eval":
            return self.__next_eval__()
    
    def __next_train__(self):
        """Get the next batch of data for training."""
        if self._batch_idx >= self.num_batches:
            raise StopIteration
        
        # Get cell lines for this batch
        start = self._batch_idx * self.batch_size
        end = start + self.batch_size
        batch_cell_lines = self.balanced_cell_lines[start:end]
        batch = {}
        
        for ab in self.keys:
            ds = self.datset_dict[ab].dataset
            x_list, y_list, feat_group_list = [], [], []
            for cl in batch_cell_lines["CellLines"]:
                idxs = (ds.metadata['CellLines'] == cl).arg_true()
                idx = random.choice(idxs)
                x, y, feat_group = ds[idx]
                x_list.append(x)
                y_list.append(y)
                feat_group_list.append(feat_group)
            
            batch[ab] = (
                torch.stack(x_list), torch.stack(y_list), feat_group_list[0]
            )
        
        self._batch_idx += 1
        return batch     
    
    def __next_eval__(self):
        """Get the next batch of data for evaluation."""
        if self._batch_idx > self.num_batches:
            raise StopIteration
        
        # Get cell lines for this batch
        start = self._batch_idx * self.batch_size
        end = start + self.batch_size
        batch_cell_lines = self.balanced_cell_lines[start:end]
        batch = {}
        
        for ab in self.keys:
            ds = self.datset_dict[ab].dataset
            x_list, y_list, cl_list, feat_group_list = [], [], [], []
            for cl in batch_cell_lines["CellLines"]:
                idxs = (ds.metadata['CellLines'] == cl).arg_true()
                idx = random.choice(idxs)
                x, y, feat_group = ds[idx]
                x_list.append(x)
                y_list.append(y)
                feat_group_list.append(feat_group)
                cl_list.append(cl)
            
            batch[ab] = (
                torch.stack(x_list), torch.stack(y_list), feat_group_list[0], cl_list
            )
        
        self._batch_idx += 1
        return batch

    def _balance_sample(self):
        """Up-sample cell lines by response category for class balance."""
        
        max_count = int(self.mutation_counts["count"].max())
        balanced_cell_lines = []
        
        for m in self.mutation_counts["Mutations"].to_list():
            meta_cl = self.cell_lines.filter(pl.col("Mutations") == m)
            n = meta_cl.height
            n_to_sample = max_count - n
            
            if n_to_sample > 0:
                idxs = np.random.choice(n, n_to_sample, replace=True)
                upsampled = [meta_cl.slice(int(i), 1) for i in idxs]
                meta_cl = pl.concat([meta_cl, pl.concat(upsampled)])

            balanced_cell_lines.append(meta_cl)
        
        return pl.concat(balanced_cell_lines)

def create_multiantibody_dataloader(
    screen: ImageScreenMultimodal,
    response: str = "Mutations",
    grouping: str = "CellLines",
    response_map: Optional[Dict[str, int]] = None,
    n_cells: int = 10,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: Optional[int] = None
) -> MultiAntibodyLoader:
    """
    Create a PyTorch DataLoader for multimodal imaging data.
    Returns a MultiAntibodyLoader that yields batches as dicts keyed by antibody.
    """
    dataloaders = {}
    
    # Initialize response key
    if response_map is None:
        merged_meta = pl.concat([s for s in screen.metadata.values()])
        response_values = merged_meta[response].unique()
        response_map = {
            resp: i for i, resp in enumerate(response_values)
        }
            
    for antibody in screen.data.keys():
        dataset = ImagingDataset(
            data=screen.data[antibody],
            metadata=screen.metadata[antibody],
            antibody=antibody,
            response=response,
            response_map=response_map,
            grouping=grouping,
            n_cells=n_cells,
            seed=seed
        )
        dataloaders[antibody] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    return MultiAntibodyLoader(dataloaders, shuffle=shuffle)


if __name__ == "__main__":
    # Example usage
    import json
    from maps.screens import ImageScreenMultimodal
    
    # Load parameters
    pdir = "/home/kkumbier/als/scripts/maps/template_analyses/params/"
    with open(pdir + "params_multimodal.json", "r") as f:
        params = json.load(f)
    
    # Create and load screen
    screen = ImageScreenMultimodal(params)
    screen.load(antibody=["FUS/EEA1"])
    screen.preprocess()

    dataloader = create_multiantibody_dataloader(screen)

    batch = dataloader.__next__()
    keys = list(batch.keys())
    batch[keys[0]][1]
    batch[keys[1]][1]
