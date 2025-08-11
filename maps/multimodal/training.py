import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from maps.multimodal.config import TrainingConfig
import wandb

def train(
    model: nn.Module, 
    dataloader: DataLoader, 
    config: TrainingConfig = TrainingConfig()):
    
    """ Configurable Adam training loop for a multi-antibody model."""
    # --- device setup ---
    device = config.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Training setup ---
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.lr, 
        betas=config.betas, 
        weight_decay=config.weight_decay
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.step_size, 
        gamma=config.gamma
    )

    criterion_cell = nn.CrossEntropyLoss()
    criterion_line = nn.CrossEntropyLoss()
    model.train()

    # --- Training loop ---
    for epoch in range(config.n_epochs):
        
        loss_cell = 0
        loss_line = 0
        
        for i, batch in enumerate(dataloader):
            if batch is None:
                continue
            
            optimizer.zero_grad()
            x_dict = {ab: batch[ab][0].to(device) for ab in batch}
            y_cell_dict = {ab: batch[ab][1].to(device) for ab in batch}
            cell_logits_dict, line_logits = model(x_dict)
            
            # Cell loss computed for each antibody
            for ab in cell_logits_dict:
                logits = cell_logits_dict[ab]  # (batch, cells, n_cell_classes)
                y_cell = y_cell_dict[ab]       # (batch,)
                y_cell_expanded = y_cell.unsqueeze(1).expand(
                    -1, logits.shape[1]).reshape(-1)
                logits_flat = logits.reshape(-1, logits.shape[-1])
                loss_cell += criterion_cell(logits_flat, y_cell_expanded)
                loss_cell = loss_cell / len(cell_logits_dict)
            
            # Cell line loss computed over all antibodies
            y_line = y_cell_dict[list(batch.keys())[0]]
            loss_line += criterion_line(line_logits, y_line)
            loss_line = loss_line / len(dataloader)
            
        # Total loss - accumulated over all cell lines
        loss = (loss_line + loss_cell)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if config.verbose:    
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}")
        if config.log:
                wandb.log({
                    "loss_line": loss_line.item(),
                    "loss_cell": loss_cell.item()
                })
            
