import pandas as pd
import torch

from maps.screens import ImageScreenMultimodal
from maps.multimodal.data_loaders import create_multiantibody_dataloader
from maps.multimodal.models import MultiModalClassifier
from maps.multimodal.training import train
import json
import wandb
from sklearn.metrics import roc_auc_score

# --- Load parameters ---
pdir = "/home/kkumbier/als/scripts/maps/template_analyses/params/"
with open(pdir + "params_multimodal.json", "r") as f:
    params = json.load(f)

# Create and load screen
screen = ImageScreenMultimodal(params)
screen.load(antibody="FUS/EEA1")
screen.preprocess()


# --- Create dataloader ---
batch_size = 6
n_cells = 50
response_map = {"WT": 0, "FUS": 1}
dataloader = create_multiantibody_dataloader(
    screen,
    batch_size=batch_size,
    n_cells=n_cells,
    response_map=response_map
)

# --- Create model ---
# Extract model configs from screen
antibody_keys = list(screen.data.keys())
antibody_feature_dims = {}
for ab in antibody_keys:
    df = screen.data[ab].drop("ID")
    antibody_feature_dims[ab] = df.shape[1]

d_model = 32
n_layers = 1

# Use first antibody for class counts
first_ab = antibody_keys[0]
n_classes = len(screen.metadata[first_ab]["Mutations"].unique())

# Initialize model
model = MultiModalClassifier(
    antibody_feature_dims=antibody_feature_dims,
    d_model=d_model,
    n_layers=n_layers,
    n_classes=n_classes
)

print("Model architecture and parameter counts:")
for name, param in model.named_parameters():
    print(f"{name}: {param.numel()} parameters")

# --- Train model ---
wandb.init(project="multimodal_training_v2", config={
    "batch_size": batch_size,
    "n_cells": n_cells,
    "d_model": d_model,
    "n_layers": n_layers,
    "n_classes": n_classes,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "n_epochs": n_epochs
})

train(model, dataloader)

# --- Evaluation ---
model.eval()
all_probs = []
all_labels = []
all_lines = []
dataloader.mode = "eval"  # Set dataloader to evaluation mode

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

all_probs = torch.cat(all_probs, dim=0)[:,1]
all_labels = torch.cat(all_labels, dim=0)

preds = pd.DataFrame({
    "CellLines": all_lines,
    "Predicted": all_probs.numpy(),
    "True": all_labels.numpy()
})

preds = preds.sort_values(
    by="Predicted", ascending=False).reset_index(drop=True)
print(preds)

auroc = roc_auc_score(all_labels.numpy(), all_probs.numpy())
print(f"AUROC: {auroc:.4f}")