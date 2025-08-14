import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from maps.multiantibody.config import TrainConfig
import wandb

def train(
    model: nn.Module, 
    dataloader: DataLoader, 
    config: TrainConfig = TrainConfig()):
    
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

    # --- Early stopping setup ---
    best_loss = float('inf')
    epochs_no_improve = 0
    patience = getattr(config, 'patience', 5)

    # --- Training loop ---
    for epoch in range(config.n_epochs):
        loss_cell = torch.tensor(0).type(torch.float).to(device)
        loss_line = torch.tensor(0).type(torch.float).to(device)

        for _, batch in enumerate(dataloader):
            if batch is None:
                continue

            optimizer.zero_grad()
            x_dict = {ab: batch[ab][0].to(device) for ab in batch}
            y_cell_dict = {ab: batch[ab][1].to(device) for ab in batch}
            cell_logits_dict, line_logits = model(x_dict)

            # Compute single cell-level loss for each antibody
            for ab in cell_logits_dict:
                logits = cell_logits_dict[ab]
                y_cell = y_cell_dict[ab]
                y_cell_expanded = y_cell.unsqueeze(1).expand(
                    -1, logits.shape[1]).reshape(-1)
                
                logits_flat = logits.reshape(-1, logits.shape[-1])
                loss_cell += criterion_cell(logits_flat, y_cell_expanded)
                
            loss_cell = loss_cell / (len(cell_logits_dict) * len(dataloader))
            
            # Cell line loss computed over all antibodies
            y_line = y_cell_dict[list(batch.keys())[0]]
            loss_line += criterion_line(line_logits, y_line)
            loss_line = loss_line / len(dataloader)

        # Total loss - accumulated over all cell lines
        loss = loss_line + loss_cell
        optimizer.step()
        scheduler.step()
        
        # Log training loss
        if config.verbose:
            print(f"Epoch {epoch+1}/{config.n_epochs}, Loss: {loss.item()}")
        if config.log and wandb.run is not None:
            wandb.log({
                "loss_line": loss_line.item(),
                "loss_cell": loss_cell.item()
            })

        # Early stopping check
        if loss.item() < best_loss - 1e-6:  # min_delta=1e-6
            best_loss = loss.item()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            if config.verbose:
                print(f"Early stopping at epoch {epoch+1}. Best loss: {best_loss}")
            break
            
