import pandas as pd
import torch

from maps.screens import ImageScreenMultiAntibody
from maps.multiantibody.data_loaders import create_multiantibody_dataloader
from maps.multiantibody.models import MultiAntibodyClassifier
from maps.multiantibody.training import train
from maps.multiantibody.config import TrainConfig, ModelConfig, DataLoaderConfig

import json
import wandb
from sklearn.metrics import roc_auc_score

# --- Initialize parameters ---
pdir = "/home/kkumbier/als/scripts/maps/template_analyses/params/"
with open(pdir + "maps_multiantibody.json", "r") as f:
    params = json.load(f)

# Create and load screen
screen = ImageScreenMultiAntibody(params)
screen.load(antibody=["FUS/EEA1", "COX IV/Galectin3/atubulin"])
screen.preprocess()

dataloader_config = DataLoaderConfig()
train_config = TrainConfig()
model_config = ModelConfig()
model_config.n_classes = 3

# --- Create dataloader ---
dataloader = create_multiantibody_dataloader(
    screen,
    **vars(dataloader_config)
)

# --- Create model ---
# Extract feature dimensions for each antibody
model_config.antibody_feature_dims = dataloader._get_feature_dims()

# Initialize model
model = MultiAntibodyClassifier(**vars(model_config))

print("Model architecture and parameter counts:")
for name, param in model.named_parameters():
    print(f"{name}: {param.numel()} parameters")

train(model, dataloader, train_config)

# --- Evaluation ---
model.eval()
all_probs = []
all_labels = []
all_lines = []
dataloader.mode = "eval"  # Set dataloader to evaluation mode
device = train_config.device

with torch.no_grad():
    for batch in dataloader:
        if batch is None:
            continue
        
        x_dict = {ab: batch[ab][0].to(device) for ab in batch}
        y_line = batch[list(batch.keys())[0]][1].to(device)
        cl = batch[list(batch.keys())[0]][-1]
        _, line_logits = model(x_dict)
        probs = torch.softmax(line_logits, dim=1)
        all_probs.append(probs.cpu())
        all_labels.append(y_line.cpu())
        all_lines.extend(cl)

all_probs = torch.cat(all_probs, dim=0)
all_labels = torch.cat(all_labels, dim=0)

preds = pd.DataFrame(all_probs.numpy())
preds.columns = [f"Class_{i}" for i in range(model_config.n_classes)]   
preds["CellLines"] = all_lines
preds["True"] = all_labels.numpy()

preds = preds.sort_values(by="Class_0", ascending=False)
print(preds)