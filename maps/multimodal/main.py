import torch
import torch.nn as nn
import torch.optim as optim
from maps.screens import ImageScreenMultimodal
from maps.multimodal.data_loaders import create_multimodal_dataloader
from maps.multimodal.models import MultiModalClassifier
import json
import wandb

# Load parameters
pdir = "/home/kkumbier/als/scripts/maps/template_analyses/params/"
with open(pdir + "params_multimodal.json", "r") as f:
    params = json.load(f)

# Create and load screen
screen = ImageScreenMultimodal(params)
screen.load(antibody="FUS/EEA1")
screen.preprocess()

# Create dataloader
batch_size = 31
n_per_group = 50
dataloader = create_multimodal_dataloader(
    screen,
    batch_size=batch_size,
    n_per_group=n_per_group
)

# --- Initialize model configs ---
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

# --- Initialize model ---
model = MultiModalClassifier(
    antibody_feature_dims=antibody_feature_dims,
    d_model=d_model,
    n_layers=n_layers,
    n_classes=n_classes
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model architecture and parameter counts:")
for name, param in model.named_parameters():
    print(f"{name}: {param.numel()} parameters")

# --- Training setup ---
optimizer = optim.Adam(
    model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5
)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

criterion_cell = nn.CrossEntropyLoss()
criterion_line = nn.CrossEntropyLoss()
model.train()

n_epochs = 50

wandb.init(project="multimodal_training", config={
    "batch_size": batch_size,
    "n_per_group": n_per_group,
    "d_model": d_model,
    "n_layers": n_layers,
    "n_classes": n_classes,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "n_epochs": n_epochs
})


# --- Training loop ---
for epoch in range(n_epochs):
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        x_dict = {ab: batch[ab][0].to(device) for ab in batch}
        y_cell_dict = {ab: batch[ab][1].to(device) for ab in batch}
        cell_logits_dict, line_logits = model(x_dict)
        
        # Cell loss
        loss_cell = 0
        for ab in cell_logits_dict:
            # For each cell line in batch, repeat label for all cells
            logits = cell_logits_dict[ab]  # (batch, cells, n_cell_classes)
            y_cell = y_cell_dict[ab]       # (batch,)
            y_cell_expanded = y_cell.unsqueeze(1).expand(-1, logits.shape[1]).reshape(-1)
            logits_flat = logits.reshape(-1, logits.shape[-1])
            loss_cell += criterion_cell(logits_flat, y_cell_expanded)
        
        # Cell line loss
        y_line = y_cell_dict[list(batch.keys())[0]]
        loss_line = criterion_line(line_logits, y_line)
        
        # Total loss
        loss = loss_line + loss_cell
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{n_epochs}], " 
              f"Loss Line: {loss_line.item():.4f}, Loss Cell: {loss_cell.item():.4f}")

        wandb.log({
            "loss_line": loss_line.item(),
            "loss_cell": loss_cell.item()
        })
