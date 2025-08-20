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
        pooled = {}

        for ab in self.antibodies:
            x = x_dict[ab]
            emb = self.encoders[ab](x)  # (batch, cells, d_model)
            cell_logits[ab] = self.cell_heads[ab](emb)
            pooled[ab] = self.pooling(emb)

        agg = self.aggregation(pooled)  # (batch, d_model * n_antibodies)
        line_logits = self.line_head(agg)

        if return_embedding:
            return cell_logits, line_logits, pooled

        return cell_logits, line_logits
    
    def group_entropy_penalty(self):
        """Compute entropy penalty for final linear head, grouped by antibody"""
        import torch.nn.functional as F
        W = self.line_head.fc.weight  # shape: [n_classes, d_model * n_antibodies]

        # Compute l2 norm of weights by antibody
        group_norms = []
        d_model = W.shape[1] // len(self.antibodies)
        for i in range(len(self.antibodies)):
            group_W = W[:, i*d_model:(i+1)*d_model]
            norm = torch.norm(group_W, p=2)
            group_norms.append(norm)
        
        group_norms = torch.stack(group_norms)  # shape: [n_antibodies]
        weights = F.softmax(group_norms, dim=0)
        entropy = -torch.sum(weights * torch.log(weights + 1e-8))
        
        return self.lambda_entropy * entropy
    
    def l2_by_antibody(self):
        """Compute L2 norm of line head parameters grouped by antibody"""
        W = self.line_head.fc.weight  # shape: [n_classes, d_model * n_antibodies]
        
        # Group weights by antibody and compute L2 norm
        l2_norms = {}
        d_model = W.shape[1] // len(self.antibodies)
        for i, ab in enumerate(self.antibodies):
            group_W = W[:, i*d_model:(i+1)*d_model]
            l2_norm = torch.norm(group_W, p=2)
            l2_norms[ab] = l2_norm
        
        return l2_norms
    
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