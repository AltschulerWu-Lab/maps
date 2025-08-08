import torch
import torch.nn as nn
import torch.optim as optim
from maps.screens import ImageScreenMultimodal
from maps.multimodal.data_loaders import create_multimodal_dataloader
from maps.multimodal.models import MultiModalClassifier
import json

# Load parameters
pdir = "/home/kkumbier/als/scripts/maps/template_analyses/params/"
with open(pdir + "params_multimodal.json", "r") as f:
    params = json.load(f)

# Create and load screen
screen = ImageScreenMultimodal(params)
screen.load(antibody=["FUS/EEA1"])
screen.preprocess()

# Create dataloader
batch_size = 24
n_per_group = 50
dataloader = create_multimodal_dataloader(
    screen,
    batch_size=batch_size,
    n_per_group=n_per_group
)

#ab = "FUS/EEA1"
#x = dataloader.dataloader_dict[ab].dataset.data.drop("ID").to_numpy()
#
#iterator = iter(dataloader)
#batch = next(iterator)
#batch = batch[list(batch.keys())[0]]

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

# --- Debugging ---
#antibody_feature_dims = {"a": 180}
model = MultiModalClassifier(
    antibody_feature_dims=antibody_feature_dims,
    d_model=d_model,
    n_layers=n_layers,
    n_classes=n_classes
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(
    model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5
)

criterion_cell = nn.CrossEntropyLoss()
criterion_line = nn.CrossEntropyLoss()
model.train()

print("Model architecture and parameter counts:")
for name, param in model.named_parameters():
    print(f"{name}: {param.numel()} parameters")


# Make temporary dataset
#import torch
#x = dataloader.dataloader_dict[ab].dataset.data.drop("ID").to_numpy() 
#x = torch.tensor(x, dtype=torch.float32)
#betas = dataloader.dataloader_dict[ab].dataset.betas
#p = torch.sigmoid(x @ betas)
#y = torch.distributions.Binomial(total_count=1, probs=p).sample()
#y = y.squeeze().long()
#
#batch_size = 64
#n_batch = x.shape[0] // batch_size
#
#n_epochs = 50
#for epoch in range(n_epochs):
#    for i in range(n_batch):
#        optimizer.zero_grad()
#        start_idx = i * batch_size
#        end_idx = start_idx + batch_size
#        xb = x[start_idx:end_idx].to(device).to(device) 
#        xb = xb.unsqueeze(1)
#        x_dict = {"a": xb}
#        y_cell_dict = {"a": y[start_idx:end_idx].to(device).to(device)}
#        cell_logits_dict, line_logits = model(x_dict)
#        
#        # Cell loss
#        loss_cell = 0
#        for ab in cell_logits_dict:
#            # For each cell line in batch, repeat label for all cells
#            logits = cell_logits_dict[ab]  # (batch, cells, n_cell_classes)
#            y_cell = y_cell_dict[ab]       # (batch,)
#            y_cell_expanded = y_cell.unsqueeze(1).expand(-1, logits.shape[1]).reshape(-1)
#            logits_flat = logits.reshape(-1, logits.shape[-1])
#            loss_cell += criterion_cell(logits_flat, y_cell_expanded)
#        
#        # Cell line loss
#        #y_line = y_cell_dict[list(batch.keys())[0]]
#        #loss_line = criterion_line(line_logits, y_line)
#        
#        # Total loss
#        loss = loss_cell
#        loss.backward()
#        optimizer.step()
#    
#    print(f"Epoch {epoch+1}/{n_epochs} - Loss: {loss.item():.4f} - Cell Loss: {loss_cell.item():.4f}")# - Line Loss: {loss_line.item():.4f}")



n_epochs = 50
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
        loss = loss_cell
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{n_epochs} - Loss: {loss.item():.4f} - Cell Loss: {loss_cell.item():.4f} - Line Loss: {loss_line.item():.4f}")

