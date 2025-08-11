from dataclasses import dataclass
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
   