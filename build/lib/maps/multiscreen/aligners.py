# Domain-Adversarial Neural Network (DANN) aligner for multiscreen
# Based on DANN.py from maps_helper
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
import numpy as np
import pandas as pd

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)
    
    def set_alpha(self, alpha):
        self.alpha = alpha

class DANNAligner(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=32, n_domains=4, num_labels=None):
        super().__init__()
        
        # Feature extractor (shared)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Gradient reversal layer
        self.grl = GradientReversalLayer()
        # Domain classifier (adversarial)
        self.domain_classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_domains)
        )
        
        # Label classifier (optional, for semi-supervised case)
        if num_labels is not None:
            self.label_classifier = nn.Sequential(
                nn.Linear(output_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, num_labels)
            )
        else:
            self.label_classifier = None
            
    def forward(self, x, alpha=1.0):
        # Extract features
        features = self.feature_extractor(x)
        
        # Domain classification (with gradient reversal)
        self.grl.set_alpha(alpha)
        reversed_features = self.grl(features)
        domain_output = self.domain_classifier(reversed_features)
        
        # Label classification (if applicable)
        label_output = None
        if self.label_classifier is not None:
            label_output = self.label_classifier(features)
        
        return features, domain_output, label_output
    
class EmptyAligner(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
    def forward(self, x, alpha=1.0):
        return x, None, None