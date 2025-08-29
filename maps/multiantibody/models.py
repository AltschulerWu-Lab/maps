import torch
import torch.nn as nn

class AntibodyEncoder(nn.Module):
    def __init__(self, in_features, d_model, n_layers):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(in_features)
        
        if n_layers > 0:
            layers = []
            for _ in range(n_layers):
                linear = nn.Linear(
                    in_features if len(layers)==0 else d_model, d_model
                )
                layers.append(linear)
                layers.append(nn.ReLU())
                layers.append(nn.LayerNorm(d_model))
                layers.append(nn.Dropout(0.3))
            self.encoder = nn.Sequential(*layers)
        else:
            self.encoder = nn.Identity()

    def forward(self, x):
        # x: (batch, cells, features)
        shape = x.shape
        x = x.view(-1, x.shape[-1])  # (batch*cells, features)
        x = self.batch_norm(x)
        x = x.view(shape)
        x = self.encoder(x)
        return x  # (batch, cells, d_model)


class CellPoolingLayer(nn.Module):
    def __init__(self, strategy='mean'):
        super().__init__()
        self.strategy = strategy

    def forward(self, x):
        # x: (batch, cells, d_model)
        if self.strategy == 'mean':
            return x.mean(dim=1)  # (batch, d_model)
        else:
            raise NotImplementedError(f"{self.strategy} not implemented.")

class AntibodyAggregationLayer(nn.Module):
    def __init__(self, strategy='concat'):
        super().__init__()
        self.strategy = strategy

    def forward(self, x_dict):
        # x_dict: {antibody: (batch, d_model)}
        if self.strategy == 'concat':
            x_list = [x_dict[k] for k in sorted(x_dict.keys())]
            return torch.cat(x_list, dim=1)  # (batch, d_model * n_antibodies)
        else:
            raise NotImplementedError(f"{self.strategy} not implemented.")

class ClassifierHead(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)
        self.batch_norm = nn.BatchNorm1d(in_features)

    def forward(self, x):
        # x: (batch, d_model * n_antibodies)
        if len(x.shape) == 3: 
            shape = x.shape
            x = x.view(-1, x.shape[-1])
            x = self.batch_norm(x)
            x = x.view(shape)
        else:
            x = self.batch_norm(x)

        logits = self.fc(x)  # (batch, n_classes)
        return logits

class MultiAntibodyClassifier(nn.Module):

    def __init__(self, antibody_feature_dims, d_model, n_layers, n_classes):
        super().__init__()
        # antibody_feature_dims: dict {antibody: in_features}
        self.antibodies = sorted(antibody_feature_dims.keys())
        
        if n_layers == 0:
            d_model = antibody_feature_dims
        else:
            d_model = {k:d_model for k in self.antibodies}
        
        self.encoders = nn.ModuleDict({
            ab: AntibodyEncoder(antibody_feature_dims[ab], d_model[ab], n_layers)
            for ab in self.antibodies
        })
        
        self.cell_heads = nn.ModuleDict({
            ab: ClassifierHead(d_model[ab], n_classes)
            for ab in self.antibodies
        })
        
        self.pooling = CellPoolingLayer()
        self.aggregation = AntibodyAggregationLayer()
        self.line_head = ClassifierHead(
            sum([v for v in d_model.values()]), n_classes
        )

        self.lambda_entropy = 1

    def forward(self, x_dict, return_embedding=False):
        # x_dict: {antibody: (batch, cells, features)}
        cell_logits = {}
        pooled_emb = {}
        cell_emb = {}

        for ab in self.antibodies:
            x = x_dict[ab]
            cell_emb[ab] = self.encoders[ab](x)  # (batch, cells, d_model)
            cell_logits[ab] = self.cell_heads[ab](cell_emb[ab])
            pooled_emb[ab] = self.pooling(cell_emb[ab])

        agg = self.aggregation(pooled_emb)  # (batch, d_model * n_antibodies)
        line_logits = self.line_head(agg)

        if return_embedding:
            return cell_logits, line_logits, cell_emb, pooled_emb,

        return cell_logits, line_logits
    
    def contrastive_loss_cell_lines(self, cell_emb, temperature=0.1):
        """
        Compute contrastive loss over the cell line axis of cell embeddings.
        """
        import torch.nn.functional as F
        
        total_loss = 0.0
        num_antibodies = 0
        
        for ab in self.antibodies:
            if ab not in cell_emb:
                continue
                
            # Normalize embeddings for cosine similarity
            emb = cell_emb[ab]  # (n_cell_lines, n_cells, d_model)
            n_cell_lines, n_cells, d_model = emb.shape
            emb_flat = emb.reshape(-1, d_model)
            emb_norm = F.normalize(emb_flat, p=2, dim=1)
            
            # Compute similarity matrix
            sim_matrix = torch.matmul(emb_norm, emb_norm.T) / temperature
            
            # Create labels: cells from same cell line should have same label
            labels = torch.arange(n_cell_lines).repeat_interleave(n_cells).to(emb.device)
            
            # Create mask for positive pairs (same cell line, different cells)
            mask = labels.unsqueeze(0) == labels.unsqueeze(1)
            
            # Remove self-similarity (diagonal)
            mask = mask & ~torch.eye(len(labels), dtype=torch.bool, device=emb.device)
            
            # Compute contrastive loss using InfoNCE
            loss = 0.0
            for i in range(len(labels)):
                # Get positive similarities (same cell line, excluding self)
                pos_mask = mask[i]
                if not pos_mask.any():
                    continue
                    
                # Numerator: sum of exp(similarities to positive pairs)
                pos_sim = sim_matrix[i][pos_mask]
                numerator = torch.exp(pos_sim).sum()
                
                # Denominator: sum of exp(similarities to all other samples)
                # Exclude self-similarity
                other_mask = torch.ones_like(mask[i])
                other_mask[i] = False
                denominator = torch.exp(sim_matrix[i][other_mask]).sum()
                
                # Add to loss: -log(numerator / denominator)
                loss += -torch.log(numerator / (denominator + 1e-8))
            
            # Average over all anchors
            loss = loss / len(labels)
            total_loss += loss
            num_antibodies += 1
        
        # Average over antibodies
        if num_antibodies > 0:
            total_loss = total_loss / num_antibodies
        
        return total_loss

    def predict_entropy_weighted(self, x=None, cell_logits=None):
        """
        Compute entropy-weighted predictions from cell logits or input x
        """
        import numpy as np
        import torch
        import torch.nn.functional as F

        antibodies = self.antibodies
        n_classes = self.line_head.fc.out_features

        if cell_logits is None:
            if x is None:
                raise ValueError("Must provide either cell_logits or x.")
            cell_logits = {}
            self.eval()
            with torch.no_grad():
                for ab in antibodies:
                    cell_emb = self.encoders[ab](x[ab])
                    cell_logits[ab] = self.cell_heads[ab](cell_emb)

        # Compute per-antibody probabilities (mean over cells)
        probs_cell = {}
        for ab in antibodies:
            # cell_logits[ab]: (batch, cells, n_classes)
            logits = cell_logits[ab]
            probs = F.softmax(logits, dim=-1)  # (batch, cells, n_classes)
            probs_mean = probs.mean(dim=1)     # (batch, n_classes)
            probs_cell[ab] = probs_mean.cpu().numpy()

        batch_size = next(iter(probs_cell.values())).shape[0]
        n_antibodies = len(antibodies)
        entropy_array = np.zeros((batch_size, n_antibodies))

        # Compute entropy for each antibody
        for i, ab in enumerate(antibodies):
            probs = probs_cell[ab]
            entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
            entropy_array[:, i] = entropy

        # Convert entropies to weights
        max_entropy = np.log(n_classes)
        weights = (max_entropy - entropy_array) / max_entropy
        weights = np.maximum(weights, 0)
        row_sums = weights.sum(axis=1, keepdims=True)
        weights = np.where(row_sums > 0, weights / row_sums, 1.0 / n_antibodies)

        # Compute weighted average predictions for all classes
        weighted_probs = np.zeros((batch_size, n_classes))
        for class_idx in range(n_classes):
            class_probs = np.zeros((batch_size, n_antibodies))
            for i, ab in enumerate(antibodies):
                class_probs[:, i] = probs_cell[ab][:, class_idx]
            weighted_probs[:, class_idx] = np.sum(weights * class_probs, axis=1)

        return weighted_probs

class LogisticClassifier(nn.Module):
    def __init__(self, antibody_feature_dims, n_classes):
        super().__init__()
        # antibody_feature_dims: dict {antibody: in_features}
        self.antibodies = sorted(antibody_feature_dims.keys())
        self.cell_heads = nn.ModuleDict({
            ab: ClassifierHead(antibody_feature_dims[ab], n_classes)
            for ab in self.antibodies
        })

    def forward(self, x_dict):
        # x_dict: {antibody: (batch, cells, features)}
        cell_logits = {}
        for ab in self.antibodies:
            cell_logits[ab] = self.cell_heads[ab](x_dict[ab])
        return cell_logits