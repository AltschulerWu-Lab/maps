# Evaluation utilities for multiscreen alignment
# Includes cosine similarity and UMAP plotting
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd

def evaluate_embedding(embedding_df_1, embedding_df_2, overlapping_cell_lines):

    # Subset the embedding array 
    filtered_embedding_1 = embedding_df_1[embedding_df_1['CellLines'].isin(overlapping_cell_lines)]
    filtered_embedding_2 = embedding_df_2[embedding_df_2['CellLines'].isin(overlapping_cell_lines)]
    # calculate mean embedding per cell line
    mean_embedding_1 = filtered_embedding_1.groupby('CellLines').mean()
    mean_embedding_2 = filtered_embedding_2.groupby('CellLines').mean()

    # Calculate cosine distance between the same cell lines in both embeddings
    distances = []
    for cell_line in overlapping_cell_lines:
        if cell_line in mean_embedding_1.index and cell_line in mean_embedding_2.index:
            # Get embeddings for the same cell line from both screens
            emb1 = mean_embedding_1.loc[cell_line].values.reshape(1, -1)
            emb2 = mean_embedding_2.loc[cell_line].values.reshape(1, -1)
            # Calculate cosine distance between the same cell line
            dist = cdist(emb1, emb2, metric='cosine')[0, 0]
            distances.append(dist)
    
    distances = np.array(distances)
    mean_distance = np.mean(distances)

    return mean_distance

def create_similarity_matrix(embedding_df, screen_info, overlapping_cell_lines):
    """
    Compute pairwise cosine distances between all unique screens in the embedding_df dataframe.
    
    Parameters
    ----------
    embedding_df : pandas.DataFrame
        Dataframe containing the high-dimensional embeddings for each cell line.
        Must have column for 'CellLines'.
    screen_info : array-like
        Vector of screen names corresponding to the embedding_df dataframe.
    overlapping_cell_lines : array-like
        Vector of cell lines that are present in all screens.
    
    Returns
    -------
    similarity_matrix : pandas.DataFrame
        A square matrix where the entry at row i and column j is the cosine distance
        between the i-th and j-th screens.
    """
    # Get unique screen names
    screen_names = np.unique(screen_info)
    pairwise_distances = {}

    # Compute pairwise distances between screens
    for i, screen1 in enumerate(screen_names):
        for j, screen2 in enumerate(screen_names):
            if i < j:
                # Subset embedding_df for each screen
                emb_df_1 = embedding_df[np.array(screen_info) == screen1]
                emb_df_2 = embedding_df[np.array(screen_info) == screen2]

                # Add screen info to each df for grouping
                emb_df_1 = emb_df_1.copy()
                emb_df_2 = emb_df_2.copy()
                emb_df_1['CellLines'] = embedding_df['CellLines']
                emb_df_2['CellLines'] = embedding_df['CellLines']

                # Calculate distance
                distance = evaluate_embedding(emb_df_1, emb_df_2, overlapping_cell_lines)

                # Store result
                pair_name = f"{screen1} vs {screen2}"
                pairwise_distances[pair_name] = distance

    # Create similarity matrix
    similarity_matrix = pd.DataFrame(index=screen_names, columns=screen_names)

    # Fill diagonal with zeros (same screen comparison)
    for screen in screen_names:
        similarity_matrix.loc[screen, screen] = 0.0

    # Fill off-diagonal elements with calculated distances
    for pair_name, distance in pairwise_distances.items():
        screen_1, screen_2 = pair_name.split(' vs ')
        similarity_matrix.loc[screen_1, screen_2] = distance
        similarity_matrix.loc[screen_2, screen_1] = distance  # Matrix is symmetric

    # Reorder similarity matrix to have the smallest distance pair first
    min_value = similarity_matrix.values[similarity_matrix.values > 0].min()  # Exclude diagonal zeros
    min_pos = np.where(similarity_matrix.values == min_value)
    min_row, min_col = min_pos[0][0], min_pos[1][0]

    screen1 = similarity_matrix.index[min_row]
    screen2 = similarity_matrix.columns[min_col]

    remaining_screens = [s for s in similarity_matrix.index if s not in [screen1, screen2]]
    new_order = [screen1, screen2] + remaining_screens

    similarity_matrix_reordered = similarity_matrix.loc[new_order, new_order]

    return similarity_matrix_reordered