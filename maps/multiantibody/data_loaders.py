"""
This module provides a PyTorch Dataset and DataLoader for handling multimodal 
imaging data from ImageScreenMultiAntibody objects. The DataLoader returns 
a dictionary of separate tensors per screen antibody.
"""
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List, Any
import random
from typing import TYPE_CHECKING
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from maps.screens import ImageScreenMultiAntibody



class ImagingDataset(Dataset):
    """
    PyTorch Dataset for imaging data from one "modality" (e.g., antibody).
    
    Creates a dataset that samples data according to response and grouping variables. Handles sampling of single cell-level features and responses by grouping variable (e.g., cell lines).
    """
    
    def __init__(
        self, 
        data: pl.DataFrame, 
        metadata: pl.DataFrame,
        antibody: str,
        response: str | List[str] = "Mutations",
        grouping: str = "CellLines", 
        response_map: Optional[Dict[str, Dict[Any, int]]] = None,
        n_cells: int = 10,
        seed: Optional[int] = None,
        scale: bool = False,
        scaler: Optional[Any] = None
    ):
        """
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
        if isinstance(response, str):
            response = [response]
        
        # Validation checks
        self._validate_inputs(response, response_map, metadata)
        
        self.data = data
        self.metadata = metadata
        self.response = response
        self.antibody = antibody
        self.grouping = grouping
        self.n_cells = n_cells
        self.response_map = response_map  
        self.scale = scale
        self.scaler = scaler

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Prepare the dataset
        self._prepare_data()
    
    def _validate_inputs(self, response: List[str], response_map: Optional[Dict[str, Dict[Any, int]]], metadata: pl.DataFrame):
        """Validate input parameters for ImagingDataset initialization."""
        # Check that all response columns exist in metadata
        metadata_columns = set(metadata.columns)
        for resp_col in response:
            if resp_col not in metadata_columns:
                raise ValueError(f"Response column '{resp_col}' not found in metadata columns: {sorted(metadata_columns)}")
        
        # Check that all keys of response_map are in the set of response values
        if response_map is not None:
            for resp_col, mapping in response_map.items():
                if resp_col not in response:
                    raise ValueError(f"Response map key '{resp_col}' not found in response list: {response}")
                
                # Get unique values from the metadata for this response column
                unique_response_values = set(metadata[resp_col].unique().to_list())
                
                # Check that all keys in the mapping exist as response values
                for map_key in mapping.keys():
                    if map_key not in unique_response_values:
                        raise ValueError(f"Response map key '{map_key}' for column '{resp_col}' not found in metadata values: {sorted(unique_response_values)}")
    
    def _make_response_map(self, resp_col: str) -> Dict[Any, int]:
        unique_vals = sorted(self.metadata[resp_col].unique())  # Sort to ensure deterministic order
        mapping = {v: i for i, v in enumerate(unique_vals)}
        return mapping
        
    def _prepare_data(self):
        # Map response values to integers, defaulting to provided key
        response_cols = self.response
        label_data = {}
        if self.response_map is None:
            self.response_map = {}
        for resp_col in response_cols:
            if resp_col in self.response_map and self.response_map[resp_col] is not None:
                mapping = self.response_map[resp_col]
            else:
                mapping = self._make_response_map(resp_col)
                self.response_map[resp_col] = mapping
            
            # Create new integer labels without modifying original metadata
            original_values = self.metadata.select(resp_col).to_series()
            mapped_values = original_values.replace(mapping).cast(pl.Int64)
            label_data[resp_col] = mapped_values
        
        # Create labels as a Polars DataFrame
        self.labels = pl.DataFrame(label_data)
        
        # Scale numeric columns if requested
        if self.scale:
            # Identify numeric columns (excluding 'ID')
            numeric_dtype = (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
            numeric_cols = [col for col, dtype in
                            zip(self.data.columns, self.data.dtypes)
                            if col != "ID" and dtype in numeric_dtype
            ]
            if self.scaler is not None:
                if not hasattr(self.scaler, 'transform'):
                    raise ValueError(
                        "Provided scaler does not have a 'transform' method."
                    )
                scaled = self.scaler.transform(
                    self.data.select(numeric_cols).to_numpy()
                )
                self.data = self.data.with_columns([
                    pl.Series(col, scaled[:, i]) 
                    for i, col in enumerate(numeric_cols)
                ])
            else:
                self.scaler = StandardScaler()
                scaled = self.scaler.fit_transform(
                    self.data.select(numeric_cols).to_numpy()
                )
                self.data = self.data.with_columns([
                    pl.Series(col, scaled[:, i])
                    for i, col in enumerate(numeric_cols)
                ])
    
    def _get_features(self, group: str):
        """Get features for a specific antibody at a given index."""
        # group = self.metadata[self.grouping].to_list()[idx]
        group_meta = self.metadata.filter(pl.col(self.grouping) == group)
        x = self.data.filter(pl.col("ID").is_in(group_meta["ID"]))

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

    def _get_response(self, group: str):
        """Get response label(s) for a specific antibody at a given index."""
        response_cols = self.response
        
        # Find the row indices where the grouping column equals the group value
        mask = self.metadata[self.grouping] == group
        group_indices = mask.arg_true()
        
        if len(group_indices) == 0:
            raise ValueError(f"No rows found for group '{group}' in column '{self.grouping}'")
        
        group_idx = group_indices[0]
        
        if len(response_cols) == 1:
            y = self.labels[response_cols[0]][group_idx]
            return torch.tensor(y, dtype=torch.long)
        else:
            ys = []
            for resp_col in response_cols:
                ys.append(self.labels[resp_col][group_idx])
            return torch.tensor(ys, dtype=torch.long)
        
    def __len__(self):
        """Return the length of the dataset."""
        return len(self.metadata)
    
    def __getitem__(self, group: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Get a single item from the dataset."""
        x, feat_group = self._get_features(group)
        y = self._get_response(group)

        return x, y, feat_group


class MultiAntibodyLoader(DataLoader):
    """
    Wraps multiple antibody-specific data loaders in a single data loader. Batches are generated as dicts keyed by antibody. Data are generated for cell lines shared across antibodies and balanced by response class.
    """
    def __init__(self, data_loader_dict, shuffle: bool=True, mode: str="train"):
        self.data_loader_dict = data_loader_dict
        self.keys = list(data_loader_dict.keys())
        
        # Find cell lines shared across antibodies
        cell_lines = {}
        for ab in self.keys:
            ds = data_loader_dict[ab].dataset
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
        self.batch_size = data_loader_dict[self.keys[0]].batch_size
        
        if self.batch_size is None:
            raise ValueError("batch_size must be specified and not None.")
        
        if self.batch_size is None:
            raise ValueError("batch_size must be specified and not None.")
        self.num_batches = len(self.balanced_cell_lines) // self.batch_size
        self._batch_idx = 0
        self.shuffle = shuffle
        self.mode = mode

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
        if self.batch_size is None:
            raise ValueError("batch_size must be specified and not None.")
        start = self._batch_idx * self.batch_size
        end = start + self.batch_size
        batch_cell_lines = self.balanced_cell_lines[start:end]
        batch = {}
        
        for ab in self.keys:
            ds = self.data_loader_dict[ab].dataset
            x_list, y_list, feat_group_list = [], [], []
            for cl in batch_cell_lines["CellLines"]:
                x, y, feat_group = ds.__getitem__(cl)
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
        if self._batch_idx >= len(self.cell_lines):
            raise StopIteration
        
        # Get cell lines for this batch
        cl = self.cell_lines["CellLines"][self._batch_idx]
        batch = {}
        
        for ab in self.keys:
            ds = self.data_loader_dict[ab].dataset
            x, y, feat_group = ds.__getitem__(cl)
            x.unsqueeze_(0)
            y.unsqueeze_(0)  # Add batch dimension for evaluation
            batch[ab] = (x, y, feat_group, cl)
        
        self._batch_idx += 1
        return batch

    def _balance_sample(self):
        """Up-sample cell lines by response category for class balance."""
        
        # Ensure we only use numeric values for max_count
        count_col = self.mutation_counts["count"]
        numeric_counts = [v for v in count_col if isinstance(v, (int, float, np.integer, np.floating))]
        if not numeric_counts:
            raise ValueError("No numeric mutation counts found for balancing.")
        max_count = int(max(numeric_counts))
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

    def _get_feature_dims(self):
        """Get the feature dimensions for each antibody."""
        feature_dims = {}
        for ab in self.keys:
            ds = self.data_loader_dict[ab].dataset.data
            feature_dims[ab] = ds.shape[1] - 1
        return feature_dims

    def _get_scalers(self):
        """Get the scalers for each antibody."""
        scalers = {}
        for ab in self.keys:
            ds = self.data_loader_dict[ab].dataset
            scalers[ab] = ds.scaler
        return scalers

def create_multiantibody_dataloader(
    screen: "ImageScreenMultiAntibody",
    response: str = "Mutations",
    grouping: str = "CellLines",
    response_map: Optional[Dict[str, int]] = None,
    mode: str = "train",
    n_cells: int = 10,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    seed: Optional[int] = None,
    scale: bool = False,
    scalers: Optional[Dict[str, Any]] = None
) -> MultiAntibodyLoader:
    """
    Create a PyTorch DataLoader for multimodal imaging data.
    Returns a MultiAntibodyLoader that yields batches as dicts keyed by antibody.
    
    Args:
        scalers: Dictionary with antibody names as keys and StandardScaler objects as values.
                If scale=True and scalers is None, new scalers will be created for each antibody.
                If scale=True and scalers is provided, the corresponding scaler will be used for each antibody.
    """
    assert screen.data is not None, "screen data is not loaded"
    assert screen.metadata is not None, "screen metadata is not loaded"
    
    dataloaders = {}
    
    # Initialize response key
    if response_map is None:
        merged_meta = pl.concat([s for s in screen.metadata.values()])
        response_values = sorted(merged_meta[response].unique())  # Sort to ensure deterministic order
        response_map = {resp: i for i, resp in enumerate(response_values)}
            
    for antibody in screen.data.keys():
        # Get the appropriate scaler for this antibody
        antibody_scaler = None
        if scale and scalers is not None:
            antibody_scaler = scalers.get(antibody, None)
        
        dataset = ImagingDataset(
            data=screen.data[antibody],
            metadata=screen.metadata[antibody],
            antibody=antibody,
            response=response,
            response_map=response_map,
            grouping=grouping,
            n_cells=n_cells,
            seed=seed,
            scale=scale,
            scaler=antibody_scaler
        )
        dataloaders[antibody] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    return MultiAntibodyLoader(dataloaders, shuffle=shuffle, mode=mode)


if __name__ == "__main__":
    # Example usage
    import json
    from maps.screens import ImageScreenMultiAntibody
    
    # Load parameters
    pdir = "/home/kkumbier/als/scripts/maps/template_analyses/params/"
    with open(pdir + "maps_multiantibody-test.json", "r") as f:
        params = json.load(f)
    
    # Create and load screen
    screen = ImageScreenMultiAntibody(params)
    screen.load(antibody=["FUS/EEA1"])
    screen.preprocess()

    dataset = ImagingDataset(
        data=screen.data["FUS/EEA1"],
        metadata=screen.metadata["FUS/EEA1"],
        antibody="FUS/EEA1",
        response=["Mutations", "CellLines"],
        grouping="CellLines",
        n_cells=10
    )
    
    # Get a sample group from the dataset to test
    sample_group = dataset.metadata[dataset.grouping].unique()[0]
    batch = dataset.__getitem__(sample_group)
    
    # Example with scalers dictionary
    scalers_dict = {"FUS/EEA1": StandardScaler()}
    dataloader = create_multiantibody_dataloader(
        screen, 
        scale=True, 
        scalers=scalers_dict
    )

    batch = dataloader.__next__()
    if batch is not None:
        keys = list(batch.keys())
        print(batch[keys[0]][1])
        if len(keys) > 1:
            print(batch[keys[1]][1])
