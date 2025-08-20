import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from maps.multiantibody.config import TrainConfig
import wandb
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
    
    # --- Training setup ---
    device = config.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # Split model parameters by hierarchy level so that only top-layer params 
    # are trained by cell line loss
    line_param_ids = set()
    for m in [model.pooling, model.aggregation, model.line_head]:
        for p in m.parameters():
            line_param_ids.add(id(p))
    
    cell_params, line_params = [], []
    for _, p in model.named_parameters(): 
        if p.requires_grad and id(p) not in line_param_ids:
            cell_params.append(p)
        elif p.requires_grad and id(p) in line_param_ids:
            line_params.append(p)
            
    optimizer_cell = optim.Adam(
        cell_params,
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay
    )
    optimizer_line = optim.Adam(
        line_params,
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay
    )

    criterion_cell = nn.CrossEntropyLoss()
    criterion_line = nn.CrossEntropyLoss()
    model.train()

    # Early stopping params
    best_loss = float('inf')
    epochs_no_improve = 0
    patience = getattr(config, 'patience', 5)

    # --- Training loop ---
    for epoch in range(config.n_epochs):
        total_loss_cell = 0
        total_loss_line = 0
        total_loss_cell_ab = {k: 0 for k in model.antibodies}
        total_acc_cell_ab = {k: 0 for k in model.antibodies}
        n_batches = 0

        for _, batch in enumerate(dataloader):
            if batch is None:
                continue

            optimizer_cell.zero_grad()
            optimizer_line.zero_grad()

            x_dict = {ab: batch[ab][0].to(device) for ab in batch}
            y_dict = {ab: batch[ab][1].to(device) for ab in batch}
            cell_logits, line_logits = model(x_dict)

            # Compute single cell-level loss for each antibody
            loss_cell = torch.tensor(0.0, device=device, requires_grad=True)
            for ab in cell_logits:
                logits = cell_logits[ab]
                y = y_dict[ab]
                y = y.unsqueeze(1).expand(-1, logits.shape[1]).reshape(-1)
                logits_flat = logits.reshape(-1, logits.shape[-1])

                ab_loss = criterion_cell(logits_flat, y)
                loss_cell = loss_cell + ab_loss / len(cell_logits)
                total_loss_cell_ab[ab] += ab_loss.item()
                
                # Compute accuracy for this antibody
                preds = torch.argmax(logits_flat, dim=1)
                acc = (preds == y).float().mean()
                total_acc_cell_ab[ab] += acc.item()
            
            total_loss_cell += loss_cell.item()

            # --- cell line level loss --- 
            y_line = y_dict[list(batch.keys())[0]]
            loss_line = criterion_line(line_logits, y_line)
            loss_line = loss_line + model.group_entropy_penalty()
            total_loss_line += loss_line.item()

            # Add l2 penalty
            l2_penalty = model.l2_by_antibody()
            for ab in l2_penalty:
                lambda_l2 = (1 - total_acc_cell_ab[ab]) * 2
                loss_line += l2_penalty[ab] * lambda_l2
            
            # --- Parameter updates ---
            loss_cell.backward(retain_graph=True)
            loss_line.backward()
            
            optimizer_cell.step()
            optimizer_line.step()
            n_batches += 1

        # Average losses over batches
        if n_batches > 0:
            total_loss_cell = total_loss_cell / n_batches
            total_loss_line = total_loss_line / n_batches
            for ab in total_loss_cell_ab:
                total_loss_cell_ab[ab] = total_loss_cell_ab[ab] / n_batches
                total_acc_cell_ab[ab] = total_acc_cell_ab[ab] / n_batches


        # Log training loss
        if config.verbose:
            print(f"Epoch {epoch+1}/{config.n_epochs}, Cell Loss: {total_loss_cell:.4f}, Line Loss: {total_loss_line:.4f}")
            for ab in model.antibodies:
                print(f"  {ab} - Cell Loss: {total_loss_cell_ab[ab]:.4f}, Cell Acc: {total_acc_cell_ab[ab]:.4f}")

        if config.log and wandb.run is not None:
            log_dict = {
                "loss_line": total_loss_line,
                "loss_cell": total_loss_cell
            }
            # Add per-antibody accuracies to wandb log
            for ab in model.antibodies:
                log_dict[f"acc_cell_{ab}"] = total_acc_cell_ab[ab]
            wandb.log(log_dict)

        # Early stopping check
        loss = total_loss_cell + total_loss_line
        if loss < best_loss - 1e-6:  # min_delta=1e-6
            best_loss = loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            if config.verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break