from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class FitConfig:
    device: str = "cuda"
    learning_rate: float = 1e-3
    alpha_schedule: str = "progressive"
    weight_decay: float = 1e-5
    n_epochs: int = 100
    lambda_domain: float = 1.0
    lambda_label: float = 1.0

@dataclass
class AlignerConfig:
    n_domains: int = 4
    hidden_dim: int = 128
    output_dim: int = 32
    
@dataclass
class DataLoaderConfig:
    label_feature: str = "CellLines"
    batch_size: int = 32
    response_map: dict = field(
        default_factory=lambda: {
            "Mutations": {"WT": 0, "FUS": 1, "SOD1": 2, "C9orf72": 3}
            }
    )