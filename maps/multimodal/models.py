import torch
import torch.nn as nn
import torch.nn.functional as F

class AntibodyEncoder(nn.Module):
    def __init__(self, in_features, d_model, n_layers):
        super().__init__()
        #self.batch_norm = nn.BatchNorm1d(in_features)
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features if len(layers)==0 else d_model, d_model))
            layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, cells, features)
        b, c, f = x.shape
        x = x.view(-1, f)  # (batch*cells, features)
        #x = self.batch_norm(x)
        x = x.view(b, c, f)
        x = self.encoder(x)
        return x  # (batch, cells, d_model)

class CellClassifierHead(nn.Module):
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (batch, cells, d_model)
        logits = self.fc(x)  # (batch, cells, n_classes)
        return logits

class CellPoolingLayer(nn.Module):
    def __init__(self, strategy='mean'):
        super().__init__()
        self.strategy = strategy

    def forward(self, x):
        # x: (batch, cells, d_model)
        if self.strategy == 'mean':
            return x.mean(dim=1)  # (batch, d_model)
        else:
            raise NotImplementedError(f"Pooling strategy {self.strategy} not implemented.")

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
            raise NotImplementedError(f"Aggregation strategy {self.strategy} not implemented.")

class CellLineClassifierHead(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        # x: (batch, d_model * n_antibodies)
        logits = self.fc(x)  # (batch, n_classes)
        return logits

class MultiModalClassifier(nn.Module):
    def __init__(self, antibody_feature_dims, d_model, n_layers, n_classes):
        super().__init__()
        # antibody_feature_dims: dict {antibody: in_features}
        self.antibodies = sorted(antibody_feature_dims.keys())
        self.encoders = nn.ModuleDict({
            ab: AntibodyEncoder(antibody_feature_dims[ab], d_model, n_layers)
            for ab in self.antibodies
        })
        self.cell_heads = nn.ModuleDict({
            ab: CellClassifierHead(d_model, n_classes)
            for ab in self.antibodies
        })
        self.pooling = CellPoolingLayer()
        self.aggregation = AntibodyAggregationLayer()
        self.line_head = CellLineClassifierHead(d_model * len(self.antibodies), n_classes)

    def forward(self, x_dict):
        # x_dict: {antibody: (batch, cells, features)}
        cell_logits = {}
        pooled = {}
        for ab in self.antibodies:
            x = x_dict[ab]
            emb = self.encoders[ab](x)  # (batch, cells, d_model)
            cell_logits[ab] = self.cell_heads[ab](emb)  # (batch, cells, n_cell_classes)
            pooled[ab] = self.pooling(emb)  # (batch, d_model)
        agg = self.aggregation(pooled)  # (batch, d_model * n_antibodies)
        line_logits = self.line_head(agg)  # (batch, n_line_classes)
        return cell_logits, line_logits
