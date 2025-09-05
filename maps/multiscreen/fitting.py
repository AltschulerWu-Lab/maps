import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import inspect

from maps.multiscreen import config
from maps.multiscreen.config import DataLoaderConfig, AlignerConfig, FitConfig
from maps.multiscreen.aligners import DANNAligner
from maps.multiscreen.data_loaders import ImagingDatasetMultiscreen

class DataTransformer:
    def __init__(self, aligner_class=DANNAligner,
                 learning_rate=0.001, n_epochs=100, device='cuda', 
                 alpha_schedule='progressive', lambda_domain=1.0, lambda_label=1.0):
        
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.alpha_schedule = alpha_schedule
        self.lambda_domain = lambda_domain
        self.lambda_label = lambda_label
        self.fitted_ = False
        self.feature_columns_ = None
        self.domain_encoding_ = None
        self.label_encoding_ = None
        self.aligner_class = aligner_class
        self.aligner_ = None
        self.device = torch.device(device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))

    def _get_alpha(self, epoch: int) -> float:
        """Get alpha value for gradient reversal based on schedule."""
        if self.alpha_schedule == 'fixed':
            return 1.0
        elif self.alpha_schedule == 'progressive':
            # Progressive schedule from DANN paper - starts slower
            p = epoch / self.n_epochs
            return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        elif self.alpha_schedule == 'warmup':
            # Warmup schedule - no adversarial training for first 25% of epochs
            warmup_epochs = int(0.25 * self.n_epochs)
            if epoch < warmup_epochs:
                return 0.0
            else:
                p = (epoch - warmup_epochs) / (self.n_epochs - warmup_epochs)
                return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        else:
            return 1.0
        
    def fit(self, df, metadf,
            dataloader_config: DataLoaderConfig = DataLoaderConfig(),
            aligner_config: AlignerConfig = AlignerConfig(), 
            fit_config: FitConfig = FitConfig()):
        # Prepare dataset and dataloader
        dataset = ImagingDatasetMultiscreen(df, metadf, label_feature=dataloader_config.label_feature, 
                                                        response_map=dataloader_config.response_map)
        self.domain_encoding_ = dataset.domain_encoding_
        self.label_encoding_ = dataset.label_encoding_
    
        # Store normalization statistics for later use
        self.feature_scaler_mean_ = dataset.feature_scaler_mean_
        self.feature_scaler_std_ = dataset.feature_scaler_std_
        dataloader = DataLoader(dataset, batch_size=dataloader_config.batch_size, shuffle=True)
        
        input_dim = len(dataset.feature_columns_)
        num_labels = len(self.label_encoding_)
        
        aligner_params = {
            'input_dim': input_dim,
            'hidden_dim': aligner_config.hidden_dim,
            'output_dim': aligner_config.output_dim,
            'n_domains': aligner_config.n_domains,
            'num_labels': num_labels
        }
        
        # Use inspect to get the aligner's constructor signature
        sig = inspect.signature(self.aligner_class.__init__)
        # Filter parameters to only those the aligner accepts
        valid_params = {k: v for k, v in aligner_params.items() 
                    if k in sig.parameters and k != 'self'}

        self.aligner_ = self.aligner_class(**valid_params).to(self.device)
        # Check if aligner has trainable parameters
        params = list(self.aligner_.parameters())
        if params:
            optimizer = optim.Adam(params, lr=fit_config.learning_rate)
        else:
            optimizer = None  # No optimizer needed for parameter-free aligners
        domain_criterion = nn.CrossEntropyLoss()
        label_criterion = nn.CrossEntropyLoss()

        print(f"Training aligner {self.aligner_class.__name__} with {aligner_config.n_domains} domains...")
        print(f"{num_labels} cell line labels")
        self.aligner_.train()

        for epoch in range(self.n_epochs):
            alpha = self._get_alpha(epoch)
            n_batches = 0
            sum_domain_loss = 0.0
            sum_label_loss = 0.0
            for batch in dataloader:
                x, d, y = batch # x = features, d = domain labels, y = class labels
                x, d, y = x.to(self.device), d.to(self.device), y.to(self.device)
                
                features, domain_pred, label_pred = self.aligner_(x, alpha=alpha)
                
                # Only compute loss and optimize if we have an optimizer
                if optimizer is not None:
                    domain_loss = domain_criterion(domain_pred, d)
                    label_loss = label_criterion(label_pred, y)
                    total_loss = self.lambda_label * label_loss + self.lambda_domain * alpha * domain_loss
                    
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                
                    n_batches += 1
                    sum_domain_loss += domain_loss.item()
                    sum_label_loss += label_loss.item()
                    
            # Print progress
            if (epoch + 1) % (max(1, self.n_epochs // 10)) == 0:
                avg_domain_loss = sum_domain_loss / n_batches if n_batches > 0 else 0.0
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Alpha: {alpha:.3f}, "
                      f"Domain Loss: {avg_domain_loss:.4f}", end="")
                
                avg_label_loss = sum_label_loss / n_batches if n_batches > 0 else 0.0
                print(f", Label Loss: {avg_label_loss:.4f}")
                
        self.fitted_ = True

    def transform(self, new_df, new_metadf,             
                  dataloader_config: DataLoaderConfig = DataLoaderConfig(),
                  aligner_config: AlignerConfig = AlignerConfig(), 
                  fit_config: FitConfig = FitConfig()):

        assert self.fitted_, "Model must be fitted before transform."

        # Pass stored mean and std to ImagingDatasetMultiscreen for normalization
        dataset = ImagingDatasetMultiscreen(
            new_df, new_metadf,
            label_feature=dataloader_config.label_feature,
            response_map=dataloader_config.response_map,
            feature_scaler_mean=self.feature_scaler_mean_,
            feature_scaler_std=self.feature_scaler_std_,
            domain_encoding=self.domain_encoding_,
            label_encoding=self.label_encoding_
        )
        dataloader = DataLoader(dataset, batch_size=dataloader_config.batch_size, shuffle=False)
        self.aligner_.eval()
        all_embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                x, _, _ = batch
                x = x.to(self.device)
                features, _, _ = self.aligner_(x, alpha=0.0)
                all_embeddings.append(features.cpu().numpy())
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        embeddings = pd.DataFrame(
            embeddings, 
            columns=[f'aligned_feature_{i}' for i in range(embeddings.shape[1])]
        )
        embeddings = pd.concat([embeddings, dataset.df[['ID', 'CellLines', 'Mutations', 'Screen']]], axis=1)
        
        return embeddings
    
    def fit_transform(self, df, metadf,
                      dataloader_config: DataLoaderConfig = DataLoaderConfig(),
                      aligner_config: AlignerConfig = AlignerConfig(), 
                      fit_config: FitConfig = FitConfig()):
        self.fit(df, metadf, dataloader_config, aligner_config, fit_config)
        return self.transform(df, metadf, dataloader_config, aligner_config, fit_config)