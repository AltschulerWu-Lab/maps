from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class TrainConfig:
    device: str = "cuda"
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-5
    step_size: int = 10
    gamma: float = 0.1
    n_epochs: int = 50
    verbose: bool = True
    log: bool = True
    patience: int = 5

@dataclass
class ModelConfig:
    n_classes: int = 2
    n_layers: int = 1
    d_model: int = 32
    antibody_feature_dims: dict = field(default_factory=dict)
    
@dataclass
class DataLoaderConfig:
    response: str = "Mutations"
    grouping: str = "CellLines"
    batch_size: int = 6
    n_cells: int = 50
    shuffle: bool = True
    mode: str = "train"
    response_map: dict = field(
        default_factory=lambda: {"WT": 0, "FUS": 1, "SOD1": 2}
    )