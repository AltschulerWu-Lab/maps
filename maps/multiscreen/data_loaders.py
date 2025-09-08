import numpy as np
import pandas as pd
import polars as pl

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field

class ImagingDatasetMultiscreen(Dataset):
    def __init__(
        self,
        df: pl.DataFrame,
        metadf: pl.DataFrame,
        response: str | List[str] = "CellLines",
        response_map: dict = field(default_factory=lambda: {
            "Mutations": {"WT": 0, "FUS": 1, "SOD1": 2, "C9orf72": 3}
        }),
        feature_scaler_mean=None,
        feature_scaler_std=None,
        domain_encoding=None,
        label_encoding=None
    ):
        if isinstance(response, str):
            response = [response]
        
        # Validation checks
        self._validate_inputs(response, response_map, metadf)

        self.df = df.to_pandas()
        self.metadf = metadf.to_pandas()
        
        self.response = response
        self.response_map = response_map

        # Storage for fitted parameters
        self.feature_columns_ = []
        self.domain_encoding_ = domain_encoding if domain_encoding is not None else {}
        self.label_encoding_ = label_encoding if label_encoding is not None else {}
        self.feature_scaler_mean_ = feature_scaler_mean
        self.feature_scaler_std_ = feature_scaler_std
        self.features = None
        self.domains = None
        self.labels = None

        self.df = self._merge_metadata_with_data()
        self.feature_columns_ = self._get_feature_columns()
        
        all_features = self.df[self.feature_columns_].values
        
        # Only fit scaler if not provided
        if self.feature_scaler_mean_ is None or self.feature_scaler_std_ is None:
            all_features = self._normalize_features(all_features, fit=True)
        
        self.features, self.domains, self.labels = self._prepare_training_data()
        self.features = self._normalize_features(self.features)
        
    def _validate_inputs(self, response: List[str], 
                         response_map: Optional[Dict[str, Dict[Any, int]]], 
                         metadf: pl.DataFrame):
        """Validate input parameters for ImagingDataset initialization."""
        # Check that all response columns exist in metadata
        metadata_columns = set(metadf.columns)
        for resp_col in response:
            if resp_col not in metadata_columns:
                raise ValueError(f"Response column '{resp_col}' not found in metadata columns: {sorted(metadata_columns)}")
        
        # Check that all keys of response_map are in the set of response values
        if response_map is not None:
            for resp_col, mapping in response_map.items():
                if resp_col not in response:
                    raise ValueError(f"Response map key '{resp_col}' not found in response list: {response}")
                
                # Get unique values from the metadata for this response column
                unique_response_values = set(metadf[resp_col].unique().to_list())

                # Check that all keys in the mapping exist as response values
                for map_key in mapping.keys():
                    if map_key not in unique_response_values:
                        raise ValueError(f"Response map key '{map_key}' for column '{resp_col}' not found in metadata values: {sorted(unique_response_values)}")
    
    def _merge_metadata_with_data(self):
        """Merge metadata with main dataframe on 'ID'."""
        if 'ID' not in self.df.columns or 'ID' not in self.metadf.columns:
            raise ValueError("Both data and metadata must contain 'ID' column for merging.")
        merged_df = pd.merge(self.df, self.metadf[['ID', 'CellLines', 'Mutations', 'Screen']], on='ID', how='left')
        if merged_df.isnull().any().any():
            raise ValueError("Merging resulted in NaN values. Check 'ID' columns for consistency.")
        return merged_df

    def _get_feature_columns(self):
        """Extract feature columns (exclude ID, CellLines, Mutation, Screen)."""
        exclude_cols = {'ID', 'CellLines', 'Mutations', 'Screen'}
        return [col for col in self.df.columns if col not in exclude_cols]

    def _prepare_training_data(self):
        """Prepare training data"""
        # Create domain mapping (screen -> domain_id)
        if not self.domain_encoding_:
            unique_screens = self.df['Screen'].unique()
            self.domain_encoding_ = {screen: i for i, screen in enumerate(unique_screens)}
        
        # Create label mapping if using labels (label -> label_id)
        if not self.label_encoding_:
            if self.response in self.response_map:
                self.label_encoding_ = self.response_map[self.response]
            else:
                unique_labels = self.df[self.response].unique()
                self.label_encoding_ = {cl: i for i, cl in enumerate(unique_labels)}

        # Prepare features, domains, and labels
        features = self.df[self.feature_columns_].values
        domains = self.df['Screen'].map(self.domain_encoding_).values
        labels = self.df[self.response].map(self.label_encoding_).values

        return features, domains, labels

    def _normalize_features(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features to zero mean and unit variance."""
        if fit or self.feature_scaler_mean_ is None or self.feature_scaler_std_ is None:
            self.feature_scaler_mean_ = np.mean(data, axis=0)
            self.feature_scaler_std_ = np.std(data, axis=0) + 1e-8
        
        return (data - self.feature_scaler_mean_) / self.feature_scaler_std_

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        d = torch.tensor(self.domains[idx])
        y = torch.tensor(self.labels[idx])
        
        return x, d, y