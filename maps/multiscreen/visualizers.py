import numpy as np
import pandas as pd

import umap
import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = {
    "WT": "#9A9A9A",
    "FUS": "#B24745" ,
    "C9orf72": "#6A6599",
    "sporadic": "#79AF97",
    "SOD1": "#00A1D5",
    "TDP43": "#DF8F44"
}
default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
                  '#bcbd22', '#17becf']


def plot_umap_embeddings(umap_df, x_limits=None, y_limits=None):
    """
    Plots UMAP embeddings colored by Screen and Mutation.

    Args:
        umap_df (pd.DataFrame): DataFrame with UMAP features in first two columns, and 'Screen', 'Mutations' columns.
        PALETTE (dict): Mapping from mutation names to colors.
    """
    # Assume first two columns are UMAP features
    umap_embeddings = umap_df.iloc[:, :2].values

    screens = umap_df['Screen'].unique()

    if x_limits is None or y_limits is None:
        # Calculate global axis limits from all UMAP embeddings
        x_min, x_max = umap_embeddings[:, 0].min(), umap_embeddings[:, 0].max()
        y_min, y_max = umap_embeddings[:, 1].min(), umap_embeddings[:, 1].max()

        # Add some padding
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        x_limits = (x_min - x_padding, x_max + x_padding)
        y_limits = (y_min - y_padding, y_max + y_padding)

    # Create figure with two subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Define plot settings for each subplot
    plot_settings = [
        {
            "ax": ax1,
            "category": "Screen",
            "items": screens,
            "color_map": {screen: default_colors[i % len(default_colors)] for i, screen in enumerate(screens)},
            "alpha": 0.25,
            "s": 5,
            "title": "UMAP Embeddings Colored by Screen"
        },
        {
            "ax": ax2,
            "category": "Mutations",
            "items": umap_df['Mutations'].unique(),
            "color_map": {mutation: PALETTE.get(mutation, 'gray') for mutation in umap_df['Mutations'].unique()},
            "alpha": 0.25,
            "s": 5,
            "title": "UMAP Embeddings Colored by Mutation"
        }
    ]

    for setting in plot_settings:
        for item in setting["items"]:
            mask = umap_df[setting["category"]] == item
            embeddings = umap_embeddings[mask]
            color = setting["color_map"][item]
            label = str(item)
            setting["ax"].scatter(
                embeddings[:, 0], embeddings[:, 1],
                label=label, alpha=setting["alpha"], s=setting["s"], color=color
            )
        setting["ax"].set_xlim(x_limits)
        setting["ax"].set_ylim(y_limits)
        setting["ax"].set_xlabel('UMAP 1')
        setting["ax"].set_ylabel('UMAP 2')
        setting["ax"].set_title(setting["title"])
        setting["ax"].legend(markerscale=3)
        setting["ax"].grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()

def plot_umap_embeddings_stratified(umap_df, x_limits=None, y_limits=None):
    """
    Plots UMAP embeddings for each screen, colored by mutation type.

    Args:
        umap_df (pd.DataFrame): DataFrame with UMAP features in first two columns, and 'Screen', 'Mutations' columns.
        PALETTE (dict): Mapping from mutation names to colors.
        x_limits (tuple, optional): Limits for x-axis.
        y_limits (tuple, optional): Limits for y-axis.
    """
    umap_embeddings = umap_df.iloc[:, :2].values
    screens = umap_df['Screen'].unique()
    n_screens = len(screens)
    n_cols = 2
    n_rows = int(np.ceil(n_screens / n_cols))
    
    if x_limits is None or y_limits is None:
        # Calculate global axis limits from all UMAP embeddings
        x_min, x_max = umap_embeddings[:, 0].min(), umap_embeddings[:, 0].max()
        y_min, y_max = umap_embeddings[:, 1].min(), umap_embeddings[:, 1].max()

        # Add some padding
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        x_limits = (x_min - x_padding, x_max + x_padding)
        y_limits = (y_min - y_padding, y_max + y_padding)

    _, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)  # flatten in case axes is 2D

    for i, screen in enumerate(screens):
        screen_mask = umap_df['Screen'] == screen
        screen_embeddings = umap_embeddings[screen_mask]
        screen_mutations = umap_df[screen_mask]['Mutations']

        for mutation in screen_mutations.unique():
            mutation_mask = screen_mutations == mutation
            color = PALETTE.get(mutation, 'gray')
            axes[i].scatter(
                screen_embeddings[mutation_mask, 0],
                screen_embeddings[mutation_mask, 1],
                label=mutation, alpha=0.25, s=5, color=color
            )

        axes[i].set_xlabel('UMAP 1')
        axes[i].set_ylabel('UMAP 2')
        axes[i].set_title(f'Screen: {str(screen)}')
        axes[i].set_xlim(x_limits)
        axes[i].set_ylim(y_limits)

        handles, labels = axes[i].get_legend_handles_labels()
        sorted_pairs = sorted(zip(handles, labels), key=lambda x: (x[1] != 'WT', x[1]))
        if sorted_pairs:
            sorted_handles, sorted_labels = zip(*sorted_pairs)
            axes[i].legend(sorted_handles, sorted_labels, markerscale=3)
        axes[i].grid(True, alpha=0.3)

    # Hide unused axes if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()