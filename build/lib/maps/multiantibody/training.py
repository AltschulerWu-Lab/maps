import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from maps.multiantibody.config import TrainConfig
from maps.multiantibody.models import MultiAntibodyClassifier
import wandb

def train(
    model: MultiAntibodyClassifier, 
    dataloader: DataLoader, 
    config: TrainConfig = TrainConfig()
):
    """Configurable Adam training loop for a multi-antibody model.
    
    Trains cell-level parameters first, then line-level parameters.
    Each phase has its own early stopping mechanism.
    """
    
    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Split model parameters by hierarchy level - cell vs patient
    line_param_ids = set()
    for module in [model.pooling, model.aggregation, model.line_head]:
        for param in module.parameters():
            line_param_ids.add(id(param))
    
    cell_params, line_params = [], []
    for _, param in model.named_parameters(): 
        if param.requires_grad and id(param) not in line_param_ids:
            cell_params.append(param)
        elif param.requires_grad and id(param) in line_param_ids:
            line_params.append(param)
    
    # Scheduler step interval (default 10)
    scheduler_step = getattr(config, 'scheduler_step', 10)

    # Create optimizers - will be updated as parameters are frozen
    def create_cell_optimizer():
        """Create optimizer for currently trainable cell parameters"""
        trainable_cell_params = []
        for _, param in model.named_parameters():
            if (param.requires_grad and 
                id(param) not in line_param_ids):
                trainable_cell_params.append(param)
        
        if trainable_cell_params:
            return optim.Adam(
                trainable_cell_params,
                lr=config.lr,
                betas=config.betas,
                weight_decay=config.weight_decay
            )
        return None

    optimizer_cell = create_cell_optimizer()
    optimizer_line = optim.Adam(
        line_params,
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay
    )

    # Learning rate schedulers
    scheduler_cell = None
    if optimizer_cell is not None:
        scheduler_cell = optim.lr_scheduler.StepLR(optimizer_cell, step_size=scheduler_step, gamma=0.5)
    
    scheduler_line = optim.lr_scheduler.StepLR(optimizer_line, step_size=scheduler_step, gamma=0.5)

    criterion_cell = nn.CrossEntropyLoss()
    criterion_line = nn.CrossEntropyLoss()
    
    # Early stopping parameters
    patience = getattr(config, 'patience', 5)
    min_delta = 1e-6
    
    if config.verbose:
        print("Starting cell-level training...")
    
    # --- Phase 1: Cell-level training ---
    # Per-antibody early stopping tracking
    best_loss_per_ab = {ab: float('inf') for ab in model.antibodies}
    epochs_no_improve_per_ab = {ab: 0 for ab in model.antibodies}
    frozen_antibodies = set()
    active_antibodies = set(model.antibodies) 
    
    model.train()
    for epoch in range(config.n_epochs):
        total_loss_cell = 0.0
        total_loss_cell_ab = {ab: 0.0 for ab in model.antibodies}
        total_acc_cell_ab = {ab: 0.0 for ab in model.antibodies}
        n_batches = 0

        for batch in dataloader:
            if batch is None:
                continue

            # Only zero gradients if we have trainable parameters
            if optimizer_cell is not None:
                optimizer_cell.zero_grad()

            # Prepare batch data
            x_dict = {ab: batch[ab][0].to(device) for ab in batch}
            y_dict = {ab: batch[ab][1].to(device) for ab in batch}
            
            # Forward pass
            cell_logits, _, cell_emb, _ = model(x_dict, return_embedding=True)

            # Compute cell-level loss for each antibody
            loss_cell = torch.tensor(0.0, device=device, requires_grad=True)
            active_antibodies = active_antibodies - frozen_antibodies
            
            for ab in active_antibodies:
                logits = cell_logits[ab]
                logits_flat = logits.reshape(-1, logits.shape[-1])

                y = y_dict[ab]
                y = y.unsqueeze(1).expand(-1, logits.shape[1]).reshape(-1)

                ab_loss = criterion_cell(logits_flat, y)
                loss_cell = loss_cell + ab_loss / len(active_antibodies)
                total_loss_cell_ab[ab] += ab_loss.item()
                
                # Compute accuracy for this antibody
                preds = torch.argmax(logits_flat, dim=1)
                acc = (preds == y).float().mean()
                total_acc_cell_ab[ab] += acc.item()

            # Add contrastive loss enforcing cell line similarity
            if config.use_contrastive_loss:
                loss_contrastive = model.contrastive_loss_cell_lines(cell_emb)
                loss_cell += loss_contrastive

            total_loss_cell += loss_cell.item()
            
            # Backward pass and optimization (only for non-frozen antibodies)
            if len(active_antibodies) > 0 and optimizer_cell is not None:
                loss_cell.backward()
                optimizer_cell.step()
            
            n_batches += 1

        # Average losses and accuracies over batches
        if n_batches > 0:
            total_loss_cell /= n_batches
            for ab in total_loss_cell_ab:
                total_loss_cell_ab[ab] /= n_batches
                total_acc_cell_ab[ab] /= n_batches

        # Per-antibody early stopping check
        newly_frozen = []
        for ab in model.antibodies:
            if ab in frozen_antibodies:
                continue
                
            current_loss = total_loss_cell_ab[ab]
            if current_loss < best_loss_per_ab[ab] - min_delta:
                best_loss_per_ab[ab] = current_loss
                epochs_no_improve_per_ab[ab] = 0
            else:
                epochs_no_improve_per_ab[ab] += 1

            # Check if this antibody should be frozen
            if epochs_no_improve_per_ab[ab] >= patience:
                frozen_antibodies.add(ab)
                newly_frozen.append(ab)
                
                # Freeze encoder and cell head parameters for this antibody
                for param in model.encoders[ab].parameters():
                    param.requires_grad = False
                for param in model.cell_heads[ab].parameters():
                    param.requires_grad = False
                
                if config.verbose:
                    print(f"  Freezing {ab} encoder at epoch {epoch+1}")

        # Recreate optimizer and scheduler if any antibodies were frozen
        if newly_frozen:
            optimizer_cell = create_cell_optimizer()
            if optimizer_cell is not None:
                scheduler_cell = optim.lr_scheduler.StepLR(optimizer_cell, step_size=scheduler_step, gamma=0.5)

        # Logging
        if config.verbose:
            active_count = len(model.antibodies) - len(frozen_antibodies)
            print(f"Cell Epoch {epoch+1}/{config.n_epochs}, "
                  f"Overall Loss: {total_loss_cell:.4f}, "
                  f"Active: {active_count}/{len(model.antibodies)}")
            
            for ab in model.antibodies:
                status = "[FROZEN]" if ab in frozen_antibodies else ""
                print(f"  {ab} - Loss: {total_loss_cell_ab[ab]:.4f}, "
                      f"Acc: {total_acc_cell_ab[ab]:.4f} {status}")

        if config.log and wandb.run is not None:
            log_dict = {"loss_cell": total_loss_cell}
            for ab in model.antibodies:
                log_dict[f"acc_cell_{ab}"] = total_acc_cell_ab[ab]
                log_dict[f"loss_cell_{ab}"] = total_loss_cell_ab[ab]
                log_dict[f"frozen_{ab}"] = 1 if ab in frozen_antibodies else 0
            log_dict["active_antibodies"] = len(model.antibodies) - len(frozen_antibodies)
            wandb.log(log_dict)

        # Step the learning rate scheduler
        if scheduler_cell is not None:
            scheduler_cell.step()

        # Stop cell training if all antibodies are frozen
        if len(frozen_antibodies) == len(model.antibodies):
            if config.verbose:
                print(f"All antibodies frozen - stopping cell training at epoch {epoch+1}")
            break
    
    if config.verbose:
        print("Starting line-level training...")
        if frozen_antibodies:
            print(f"Unfreezing all antibody encoders for line training...")
    
    # Unfreeze all parameters for line training
    for ab in model.antibodies:
        for param in model.encoders[ab].parameters():
            param.requires_grad = True
        for param in model.cell_heads[ab].parameters():
            param.requires_grad = True
    
    # --- Phase 2: Line-level training --- 
    best_line_loss = float('inf')
    epochs_no_improve_line = 0
    
    model.train()
    for epoch in range(config.n_epochs):
        total_loss_line = 0.0
        total_acc_cell_ab = {ab: 0.0 for ab in model.antibodies}
        n_batches = 0

        for batch in dataloader:
            if batch is None:
                continue

            optimizer_line.zero_grad()

            # Prepare batch data
            x_dict = {ab: batch[ab][0].to(device) for ab in batch}
            y_dict = {ab: batch[ab][1].to(device) for ab in batch}
            
            # Forward pass
            cell_logits, line_logits = model(x_dict)

            # Compute cell accuracies for L2 penalty weighting
            for ab in cell_logits:
                logits = cell_logits[ab]
                y = y_dict[ab]
                y = y.unsqueeze(1).expand(-1, logits.shape[1]).reshape(-1)
                logits_flat = logits.reshape(-1, logits.shape[-1])
                
                preds = torch.argmax(logits_flat, dim=1)
                acc = (preds == y).float().mean()
                total_acc_cell_ab[ab] += acc.item()

            # Line-level loss
            y_line = y_dict[list(batch.keys())[0]]
            loss_line = criterion_line(line_logits, y_line)
            total_loss_line += loss_line.item()

            # Backward pass and optimization
            loss_line.backward()
            optimizer_line.step()
            n_batches += 1

        # Average losses over batches
        if n_batches > 0:
            total_loss_line /= n_batches
            for ab in total_acc_cell_ab:
                total_acc_cell_ab[ab] /= n_batches

        # Logging
        if config.verbose:
            print(f"Line Epoch {epoch+1}/{config.n_epochs}, Loss: {total_loss_line:.4f}")

        if config.log and wandb.run is not None:
            wandb.log({"loss_line": total_loss_line})

        # Step the learning rate scheduler
        scheduler_line.step()

        # Early stopping check for line training
        if total_loss_line < best_line_loss - min_delta:
            best_line_loss = total_loss_line
            epochs_no_improve_line = 0
        else:
            epochs_no_improve_line += 1

        if epochs_no_improve_line >= patience:
            if config.verbose:
                print(f"Early stopping line training at epoch {epoch+1}")
            break
    
    if config.verbose:
        print("Training completed!")