import numpy as np
import torch
import pandas as pd

def eval(model, dataloader):
    model.eval()
    probs_line = []
    probs_entropy = []
    probs_cell = {}
    labels = []
    cell_lines = []
    embeddings = {}

    device = next(model.parameters()).device  # Get device from model parameters
    dataloader.batch_size
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            
            x_dict = {ab: batch[ab][0].to(device) for ab in batch}
            y_line = batch[list(batch.keys())[0]][1].to(device)
            cell_lines.extend([batch[list(batch.keys())[0]][-1]])

            cell_logits, line_logits, embs, _ = model(
                x_dict, return_embedding=True
            )        
            
            labels.append(y_line.cpu())
            probs_line.append(torch.softmax(line_logits, dim=-1).cpu())

            for ab in cell_logits:
                if ab not in probs_cell:
                    probs_cell[ab] = []        
                probs_cell[ab].append(torch.softmax(cell_logits[ab], dim=-1).cpu())
            
            for ab in embs:
                if not ab in embeddings:
                    embeddings[ab] = []
                embeddings[ab].append(embs[ab].cpu())
                
            probs_entropy.append(
                model.predict_entropy_weighted(cell_logits=cell_logits)
            )

    # Merge outputs over cell lines
    probs_line = torch.cat(probs_line)
    probs_entropy = np.concatenate(probs_entropy)
               
    labels = torch.cat(labels)
    probs_cell = {k: torch.cat(v).mean(dim=1) for k, v in probs_cell.items()}
    embeddings = {k: torch.cat(v) for k, v in embeddings.items()}
    
    # Create DataFrame with aggregate class probabilities from probs_line
    df = pd.DataFrame(probs_line.numpy())
    df.columns = [f"class_{i}_agg" for i in df.columns]

    df_entropy = pd.DataFrame(probs_entropy)
    df_entropy.columns = [f"class_{i}_entropy" for i in df_entropy.columns]
    df = pd.concat([df, df_entropy], axis=1)

    # Add per-antibody class probabilities from probs_cell
    for ab in probs_cell:
        dfab = pd.DataFrame(probs_cell[ab].numpy())
        dfab.columns = [f"class_{i}_{ab}" for i in dfab.columns]
        df = pd.concat([df, dfab], axis=1)

    df["CellLines"] = cell_lines
    df["True"] = labels.numpy()
    return df, embeddings