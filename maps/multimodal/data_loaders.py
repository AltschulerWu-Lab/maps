"""
PyTorch DataLoader for ImageScreenMultimodal data.

This module provides a PyTorch Dataset and DataLoader for handling multimodal 
imaging data from ImageScreenMultimodal objects. The DataLoader returns 
separate tensors for each antibody as dictionaries rather than combining 
all antibodies into a single tensor.
"""

# TODO: cell lines are comparable between biomarkers, but that drops class balance - move class balancing to data loader


import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List
import random
from maps.screens import ImageScreenMultimodal


class AntibodyDataset(Dataset):
    """
    PyTorch Dataset for multimodal imaging data.
    
    Takes an ImageScreenMultimodal object and creates a dataset that samples
    data according to response and grouping variables. Returns separate tensors
    for each antibody rather than combining them.
    """
    
    def __init__(
        self, 
        data: pl.DataFrame, 
        metadata: pl.DataFrame,
        antibody: str,
        normalize: bool = True,
        response: str = "Mutations",
        grouping: str = "CellLines", 
        response_map: Optional[Dict[str, int]] = None,
        n_per_group: int = 10,
        seed: Optional[int] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            screen: ImageScreenMultimodal object with loaded and preprocessed 
            data
            antibody: Antibody to use for the dataset
            response: Column name for response variable (default: "Mutations")
            grouping: Column name for grouping variable (default: "cell_lines")
            n_per_group: Number of samples to sample per group
            seed: Random seed for reproducible sampling
        """
        self.data = data
        self.metadata = metadata
        self.response = response
        self.antibody = antibody
        self.grouping = grouping
        self.n_per_group = n_per_group
        self.normalize = normalize
        self.response_map = response_map  
        self.betas = None
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Prepare the dataset
        self._prepare_data()
        
    def _prepare_data(self):
        """
        Prepare the data by sampling according to response and grouping variables.
        """
        
        # Get unique response values across all antibodies to ensure consistency
        balanced_meta = self._balance_samples(self.metadata)
        
        # Map response strings to numeric labels for each antibody
        if self.response_map is None:
            response_values = balanced_meta[self.response].unique()
            self.response_map = {
                resp: i for i, resp in enumerate(response_values)
            }
        
        self.metadata = balanced_meta.with_columns(
            pl.col(self.response).replace(self.response_map).alias("Label")
        )
        
        self.metadata = self.metadata.with_columns(
            pl.col("Label").cast(pl.Int64)
        )
        
        # center and scale the data for selected antibody
        if self.normalize:
            numeric_cols = [c for c in self.data.columns if c != "ID"]
            self.data = self.data.with_columns([
                ((pl.col(c) - pl.col(c).mean()) / (pl.col(c).std() + 1e-6)).alias(c)
                for c in numeric_cols
            ])
        
    def _balance_samples(self, metadata: pl.DataFrame) -> pl.DataFrame:
        """
        Up-sample metadata by response category to ensure class balance.
        """
        counts = metadata.group_by(self.response).agg(
            [pl.count("ID").alias("count")])
        
        max_count = int(counts["count"].max())
        balanced_meta = []
        
        for resp in counts[self.response].to_list():
            meta_resp = metadata.filter(pl.col(self.response) == resp)
            n = meta_resp.height
            n_to_sample = max_count - n
            
            if n_to_sample > 0:
                idx = np.random.choice(n, n_to_sample, replace=True)
                upsampled = [meta_resp.slice(i, 1) for i in idx]
                meta_resp = pl.concat([meta_resp, pl.concat(upsampled)])

            balanced_meta.append(meta_resp)
        return pl.concat(balanced_meta)
    
    def _get_features(self, idx: int):
        """
        Get features for a specific antibody at a given index.
        """
        id_select = self.metadata["ID"].to_list()[idx]
        x = self.data.filter(pl.col("ID") == id_select)

        # Sample self.n_per_group entries from x (with replacement if needed)
        # Use deterministic sampling based on idx to ensure consistency
        #np.random.seed(idx + 42)  # Add constant to avoid seed=0
        n_rows = x.height
        if n_rows >= self.n_per_group:
            idxs = np.random.choice(n_rows, self.n_per_group, replace=False)
        else:
            idxs = np.random.choice(n_rows, self.n_per_group, replace=True)
        
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
        """
        Get a single item from the dataset.
        """
        x, feat_group = self._get_features(idx)
        y = self._get_response(idx)

        return x, y, feat_group


class MultiAntibodyLoader:
    """
    Wraps multiple antibody-specific dataloaders and yields batches as dicts keyed by antibody.
    """
    def __init__(self, dataloader_dict, shuffle: bool = True):
        self.dataloader_dict = dataloader_dict
        self.keys = list(dataloader_dict.keys())
        
        # Get cell lines for each antibody
        self.cell_lines_per_ab = {}
        for ab in self.keys:
            ds = dataloader_dict[ab].dataset
            self.cell_lines_per_ab[ab] = set(ds.metadata['CellLines'].to_list())
        
        # Find intersection of cell lines
        self.common_cell_lines = sorted(
            set.intersection(*self.cell_lines_per_ab.values())
        )
        
        self.batch_size = dataloader_dict[self.keys[0]].batch_size
        self.num_batches = len(self.common_cell_lines) // self.batch_size
        self._batch_idx = 0
        self.shuffle = shuffle
        self._cell_lines_order = self.common_cell_lines.copy()

    def __iter__(self):
        self._batch_idx = 0
        # Shuffle cell lines at the start of each epoch if enabled
        if self.shuffle:
            self._cell_lines_order = self.common_cell_lines.copy()
            random.shuffle(self._cell_lines_order)
        else:
            self._cell_lines_order = self.common_cell_lines.copy()
        return self

    def __len__(self):
        return self.num_batches
    
    def __next__(self):
        # Standard PyTorch DataLoader behavior: raise StopIteration at end of epoch
        if self._batch_idx >= self.num_batches:
            raise StopIteration
        
        # Get cell lines for this batch
        start = self._batch_idx * self.batch_size
        end = start + self.batch_size
        batch_cell_lines = self._cell_lines_order[start:end]
        batch = {}
        for ab in self.keys:
            ds = self.dataloader_dict[ab].dataset
            x_list, y_list, feat_group_list = [], [], []
            for cl in batch_cell_lines:
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

def create_multimodal_dataloader(
    screen: ImageScreenMultimodal,
    response: str = "Mutations",
    grouping: str = "CellLines",
    n_per_group: int = 10,
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
    merged_meta = pl.concat([s for s in screen.metadata.values()])
    response_values = merged_meta[response].unique()
    response_map = {
        resp: i for i, resp in enumerate(response_values)
    }
            
    for antibody in screen.data.keys():
        dataset = AntibodyDataset(
            data=screen.data[antibody],
            metadata=screen.metadata[antibody],
            antibody=antibody,
            response=response,
            response_map=response_map,
            grouping=grouping,
            n_per_group=n_per_group,
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
    screen.load(antibody=["FUS/EEA1", "COX IV/Galectin3/atubulin"])
    screen.preprocess()
    
    # Create dataset and dataloader
    dataset = AntibodyDataset(screen, n_per_group=10, antibody="FUS/EEA1")
    sample = dataset.__getitem__(0)
    
    dataloader = create_multimodal_dataloader(
        screen, batch_size=10, n_per_group=10
    )
    
    batch = dataloader.__next__()
    keys = list(batch.keys())
    batch[keys[0]][1]
    batch[keys[1]][1]
