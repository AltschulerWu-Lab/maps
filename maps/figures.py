"""
Updated figure generation functions for i-MAP analysis.

This module provides plotting functions that operate on dataframes with i-MAP scores
in a standard format:
- 'CellLines': observation cell line for each row
- 'True': true class integer value
- 'Class_0', 'Class_1', ..., 'Class_k': class predictions/probabilities

Functions may accept additional metadata mappings (mutations, age, etc.) as needed.
"""

from typing import Dict, List, Tuple, Optional, Union, Mapping, cast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mpl_cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from matplotlib import patches as mpatches

# Default color palette for mutations
PALETTE = {
    "WT": "#9A9A9A",
    "FUS": "#B24745",
    "C9orf72": "#6A6599",
    "sporadic": "#79AF97",
    "SOD1": "#00A1D5",
    "TDP43": "#DF8F44"
}


# ============================================================================
# Helper Functions
# ============================================================================

def _ensure_mutations_column(
    df: pd.DataFrame,
    cellline_to_mutation: Union[Dict[str, str], pd.Series]
) -> pd.DataFrame:
    """Ensure a 'Mutations' column exists by mapping from CellLines."""
    if 'Mutations' not in df.columns:
        df = df.copy()
        df['Mutations'] = df['CellLines'].map(cellline_to_mutation)
    return df


def _convert_to_long_format(
    df: pd.DataFrame,
    class_name_map: Optional[Dict[int, str]] = None
) -> pd.DataFrame:
    """
    Convert wide format (Class_0, Class_1, ...) to long format with ModelGenetics column.
    
    Args:
        df: DataFrame with columns CellLines, True, Class_0, Class_1, ...
        class_name_map: Optional mapping from class index to class name
        
    Returns:
        DataFrame in long format with ModelGenetics and Predictions columns
    """
    # Find all Class_* columns
    class_cols = [col for col in df.columns if col.startswith('Class_')]
    
    # Melt to long format
    id_vars = ['CellLines', 'True']
    if 'Mutations' in df.columns:
        id_vars.append('Mutations')
    
    long_df = df.melt(
        id_vars=id_vars,
        value_vars=class_cols,
        var_name='ModelGenetics',
        value_name='Predictions'
    )
    
    # Extract class index from column name
    long_df['ClassIndex'] = long_df['ModelGenetics'].str.replace('Class_', '').astype(int)
    
    # Map to class names if provided
    if class_name_map is not None:
        long_df['ModelGenetics'] = long_df['ClassIndex'].map(class_name_map)
    else:
        long_df['ModelGenetics'] = long_df['ClassIndex']
    
    # Rename Predictions to Class_1 for consistency with existing code
    long_df = long_df.rename(columns={'Predictions': 'Class_1'})
    
    return long_df


def _order_cell_lines_by_group(
    predictions: pd.DataFrame,
    cellline_to_mutation: Union[Dict[str, str], pd.Series],
) -> Tuple[List[str], Dict[str, str]]:
    """
    Produce a stable ordering of cell lines by mutation groups for heatmaps/plots.
    
    - Within each mutation group g (excluding WT/sporadic), order by mean Class_1 
      for rows where ModelGenetics==g.
    - For WT/sporadic, order by overall mean Class_1 across all ModelGenetics.
    
    Args:
        predictions: DataFrame with columns CellLines, Class_1, ModelGenetics
        cellline_to_mutation: Mapping from cell line names to mutation labels
        
    Returns:
        Tuple of (ordered_celllines, cellline_to_group)
    """
    df = _ensure_mutations_column(predictions, cellline_to_mutation)
    
    # If ModelGenetics is missing (single-model predictions), create a dummy level
    if 'ModelGenetics' not in df.columns:
        df = df.copy()
        df['ModelGenetics'] = 'Model'
    
    # Mean per (Mutations, CellLines)
    cellline_means = (
        df.groupby(['Mutations', 'CellLines'])['Class_1']
        .mean()
        .reset_index()
    )
    
    all_groups = cellline_means['Mutations'].dropna().unique().tolist()
    model_groups = [g for g in all_groups if g not in ['WT', 'sporadic']]
    
    ordered_celllines: List[str] = []
    cellline_to_group: Dict[str, str] = {}
    
    for mutation in all_groups:
        group = cellline_means[cellline_means['Mutations'] == mutation]
        if mutation in model_groups:
            # For model groups, order by mean Class_1 when ModelGenetics matches
            relevant_cells = df[(df['ModelGenetics'] == mutation) & (df['Mutations'] == mutation)]
            ordered = (
                relevant_cells.groupby('CellLines')['Class_1']
                .mean()
                .reset_index()
                .sort_values('Class_1', ascending=False)['CellLines']
                .tolist()
            )
            # Add any missing cell lines from this group
            missing = [c for c in group['CellLines'] if c not in ordered]
            ordered += missing
        else:
            # For WT/sporadic, order by overall mean Class_1
            mean_preds = df[df['CellLines'].isin(group['CellLines'])]
            ordered = (
                mean_preds.groupby('CellLines')['Class_1']
                .mean()
                .reset_index()
                .sort_values('Class_1', ascending=False)['CellLines']
                .tolist()
            )
        ordered_celllines.extend(ordered)
        for cl in ordered:
            cellline_to_group[cl] = mutation
    
    # Deduplicate preserving order
    seen = set()
    ordered_celllines_unique = []
    for cl in ordered_celllines:
        if cl not in seen:
            seen.add(cl)
            ordered_celllines_unique.append(cl)
    
    return ordered_celllines_unique, cellline_to_group


def _prepare_barplot_data(
    df: pd.DataFrame,
    class_name_map: Dict[int, str]
) -> pd.DataFrame:
    """
    Convert standard format to barplot format.
    
    Args:
        df: DataFrame with CellLines, True, Class_0, Class_1, ...
        class_name_map: Mapping from class index to class name
        
    Returns:
        DataFrame with columns: CellLine, TrueClass, PredictedClass, Probability, etc.
    """
    prob_cols = [col for col in df.columns if col.startswith('Class_')]
    plot_data_list = []
    
    # Check if 'True' column exists, if not infer from highest probability
    has_true_col = 'True' in df.columns
    
    for _, row in df.iterrows():
        cellline = row['CellLines']
        if has_true_col:
            true_class = int(row['True'])
        else:
            # Infer true class from highest probability
            probs = [row[col] for col in prob_cols]
            true_class = int(np.argmax(probs))
        
        # Get probabilities for all classes
        probs = [row[col] for col in prob_cols]
        
        # Create a row for each class probability
        for class_idx, prob in enumerate(probs):
            plot_data_list.append({
                'CellLine': cellline,
                'TrueClass': class_name_map.get(true_class, f'Class_{true_class}'),
                'PredictedClass': class_name_map.get(class_idx, f'Class_{class_idx}'),
                'Probability': prob,
                'TrueClassProb': probs[true_class],
                'ClassIndex': class_idx,
            })
    
    return pd.DataFrame(plot_data_list)


# ============================================================================
# Main Plotting Functions
# ============================================================================

def plot_imap_heatmap(
    predictions: pd.DataFrame,
    cellline_to_mutation: Union[Dict[str, str], pd.Series],
    class_name_map: Optional[Dict[int, str]] = None,
    cellline_to_age: Optional[Union[Dict[str, float], pd.Series]] = None,
    row_scale: bool = True,
    antibodies_label: Optional[str] = None,
    palette: Optional[Mapping[str, str]] = None,
    figsize: Tuple[int, int] = (12, 8),
):
    """
    Create heatmap of i-MAP scores across CellLines (x) and ModelGenetics (y).
    
    Args:
        predictions: DataFrame with columns CellLines, True, Class_0, Class_1, ..., Class_k
        cellline_to_mutation: Mapping from cell line names to mutation labels
        class_name_map: Optional mapping from class index to class name (e.g., {0: 'WT', 1: 'FUS'})
                       Maps Class_i column indices to their corresponding mutation names
        cellline_to_age: Optional mapping from cell line names to age values
        row_scale: If True, scale each ModelGenetics row independently
        antibodies_label: Optional label to append to ModelGenetics names
        palette: Optional color palette for mutations
        figsize: Figure size as (width, height)
        
    Returns:
        matplotlib Figure object
    """
    palette = cast(Mapping[str, str], PALETTE if palette is None else palette)
    
    # Convert to long format if needed
    if 'ModelGenetics' not in predictions.columns:
        df = _convert_to_long_format(predictions, class_name_map)
        df = _ensure_mutations_column(df, cellline_to_mutation)
    else:
        df = predictions.copy()
        df = _ensure_mutations_column(df, cellline_to_mutation)
    
    # Order cell lines by groups
    ordered_celllines, cellline_to_group = _order_cell_lines_by_group(df, cellline_to_mutation)
    
    plot_df = df.copy()
    
    # Optional row scaling by ModelGenetics
    if row_scale:
        def _scale_row(g):
            min_val = g['Class_1'].min()
            max_val = g['Class_1'].max()
            if max_val > min_val:
                g['Class_1_scaled'] = (g['Class_1'] - min_val) / (max_val - min_val)
            else:
                g['Class_1_scaled'] = 0.5
            return g
        plot_df = plot_df.groupby('ModelGenetics', group_keys=False).apply(_scale_row)
        fill_col = 'Class_1_scaled'
    else:
        fill_col = 'Class_1'
    
    if antibodies_label:
        plot_df = plot_df.copy()
        plot_df['ModelGenetics'] = plot_df['ModelGenetics'].astype(str) + f"\n({antibodies_label})"
    
    plot_df['CellLines'] = pd.Categorical(
        plot_df['CellLines'], 
        categories=ordered_celllines, 
        ordered=True
    )
    
    # Pivot for heatmap
    pivot = plot_df.pivot_table(
        index='ModelGenetics',
        columns='CellLines',
        values=fill_col,
        aggfunc='mean'
    )
    
    # Continuous low-mid-high colormap
    cmap = LinearSegmentedColormap.from_list('imap_div', ['#014B8D', '#000000', '#D40C11'], N=256)
    
    # Layout: 3 rows (mutation strip, age strip, heatmap), share x
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, width_ratios=[1, 0.05], height_ratios=[0.3, 0.3, 10], figure=fig)
    ax_mut = fig.add_subplot(gs[0, 0])
    ax_age = fig.add_subplot(gs[1, 0], sharex=ax_mut)
    ax = fig.add_subplot(gs[2, 0], sharex=ax_mut)
    cbar_ax = fig.add_subplot(gs[2, 1])
    
    # Heatmap
    sns.heatmap(
        pivot,
        cmap=cmap,
        vmin=0,
        vmax=1,
        ax=ax,
        cbar=True,
        cbar_ax=cbar_ax,
    )
    ax.set_xlabel('')
    ax.set_ylabel('Model Genetics')
    
    # Compute group boundaries and centers
    x_labels = list(pivot.columns)
    group_labels = [cellline_to_group.get(cl, '') for cl in x_labels]
    unique_groups: List[str] = []
    group_positions: List[int] = []
    for i, g in enumerate(group_labels):
        if i == 0 or g != group_labels[i-1]:
            unique_groups.append(g)
            group_positions.append(i)
    group_positions.append(len(group_labels))
    group_centers = [
        (group_positions[i] + group_positions[i+1] - 1) / 2 
        for i in range(len(unique_groups))
    ]
    
    # Mutation strip (top)
    ax_mut.set_ylim(0, 1)
    ax_mut.set_yticks([])
    ax_mut.set_xticks([])
    for i, cl in enumerate(x_labels):
        color = palette.get(cellline_to_group.get(cl, ''), '#ffffff')
        ax_mut.add_patch(mpatches.Rectangle((i, 0), 1, 1, color=color, clip_on=False))
    
    # Group labels centered
    for center, gname in zip(group_centers, unique_groups):
        ax_mut.text(
            center + 0.5, 0.5, gname,
            ha='center', va='center',
            fontsize=10, color='black', fontweight='bold'
        )
    
    # Group separators
    for pos in group_positions[1:-1]:
        for a in (ax_mut, ax_age, ax):
            a.axvline(pos, color='grey', linestyle='--', linewidth=1.0, zorder=2)
    
    # Age strip (below mutation strip)
    ax_age.set_ylim(0, 1)
    ax_age.set_yticks([])
    ax_age.set_xticks(range(len(x_labels)))
    ax_age.set_xticklabels([])
    
    valid_ages: List[float] = []
    age_vals: List[Optional[float]] = []
    if cellline_to_age is not None:
        for cl in x_labels:
            age = cellline_to_age.get(cl, None)
            age_vals.append(age)
            if isinstance(age, (int, float)) and not pd.isna(age):
                valid_ages.append(float(age))
    
    if cellline_to_age is not None and len(valid_ages) > 0:
        norm = Normalize(vmin=min(valid_ages), vmax=max(valid_ages))
        age_cmap = mpl_cm.get_cmap('Oranges')
        for i, age in enumerate(age_vals):
            if isinstance(age, (int, float)) and not pd.isna(age):
                color = age_cmap(norm(age))
            else:
                color = (0.7, 0.7, 0.7)
            ax_age.add_patch(mpatches.Rectangle((i, 0), 1, 1, color=color, clip_on=False))
        
        # Age colorbar and missing legend
        bbox = cbar_ax.get_position()
        age_cax = fig.add_axes([bbox.x1 + 0.02, bbox.y0 + 0.5*(bbox.height-0.2), 0.02, 0.2])
        sm = mpl_cm.ScalarMappable(cmap=age_cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=age_cax)
        
        # Missing patch legend - position below the age colorbar
        age_box = age_cax.get_position()
        age_legend = fig.legend(
            handles=[Patch(facecolor=(0.7, 0.7, 0.7), label='Missing')],
            title='', title_fontsize=11,
            loc='upper left',
            bbox_to_anchor=(age_box.x0, age_box.y0 - 0.05),
            bbox_transform=fig.transFigure,
            frameon=True, fontsize=10,
        )
    
    # Improve spacing
    fig.subplots_adjust(top=0.92, right=0.85, hspace=0.05)
    return fig


def plot_imap_scatter(
    predictions: pd.DataFrame,
    cellline_to_mutation: Union[Dict[str, str], pd.Series],
    class_name_map: Optional[Dict[int, str]] = None,
    palette: Optional[Mapping[str, str]] = None,
    figsize: Tuple[int, int] = (8, 6),
):
    """
    Create scatter plot of i-MAP scores vs model genetics with custom positioning and alpha.
    
    Args:
        predictions: DataFrame with columns CellLines, True, Class_0, Class_1, ..., Class_k
        cellline_to_mutation: Mapping from cell line names to mutation labels
        class_name_map: Optional mapping from class index to class name
                       Maps Class_i column indices to their corresponding mutation names
        palette: Optional color palette for mutations
        figsize: Figure size as (width, height)
        
    Returns:
        matplotlib Figure object
    """
    palette = cast(Mapping[str, str], PALETTE if palette is None else palette)
    
    # Convert to long format if needed
    if 'ModelGenetics' not in predictions.columns:
        df = _convert_to_long_format(predictions, class_name_map)
        df = _ensure_mutations_column(df, cellline_to_mutation)
    else:
        df = predictions.copy()
        df = _ensure_mutations_column(df, cellline_to_mutation)
    
    scatter_df = df.copy()
    scatter_df['MutationColor'] = scatter_df['Mutations'].map(palette)
    scatter_df['i-MAP score'] = scatter_df['Class_1']
    
    model_order = {gen: i for i, gen in enumerate(scatter_df['ModelGenetics'].unique())}
    scatter_df['y_numeric'] = scatter_df['ModelGenetics'].map(model_order)
    
    all_mutations = [m for m in scatter_df['Mutations'].unique() if m not in ['WT', 'sporadic']]
    
    def get_mutation_position(row):
        model_gen = row['ModelGenetics']
        mutation = row['Mutations']
        if mutation == 'WT':
            return 0.0
        elif mutation == model_gen:
            return 0.2
        elif mutation == 'sporadic':
            return 0.4
        else:
            other_genetics = [m for m in all_mutations if m != model_gen]
            if mutation in other_genetics:
                idx = other_genetics.index(mutation)
                return 1.0 + (idx * 0.3)
            else:
                return 1.0
    
    scatter_df['mutation_position'] = scatter_df.apply(get_mutation_position, axis=1)
    scatter_df['mutation_jitter'] = scatter_df['mutation_position'] + np.random.uniform(
        -0.05, 0.05, size=len(scatter_df)
    )
    scatter_df['y_jitter'] = scatter_df['y_numeric'] * 0.8 + scatter_df['mutation_jitter'] * 0.25
    
    def get_alpha(row):
        mutation = row['Mutations']
        if mutation in ['WT', 'sporadic'] or mutation == row['ModelGenetics']:
            return 0.8
        else:
            return 0.3
    scatter_df['alpha'] = scatter_df.apply(get_alpha, axis=1)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    for mut, sub in scatter_df.groupby('Mutations'):
        base_colors = sub['MutationColor'].tolist()
        alphas = sub['alpha'].tolist()
        rgba_colors = [mcolors.to_rgba(c, alpha=a) for c, a in zip(base_colors, alphas)]
        ax.scatter(sub['i-MAP score'], sub['y_jitter'], label=mut, c=rgba_colors, s=30)
    
    ax.set_xlim(0, 1)
    ax.set_yticks([i * 0.8 for i in model_order.values()])
    ax.set_yticklabels(list(model_order.keys()))
    ax.set_xlabel('i-MAP score')
    ax.set_ylabel('Model Genetics')
    ax.legend(title='', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    return fig


def plot_imap_barplot(
    predictions: pd.DataFrame,
    class_name_map: Dict[int, str],
    cellline_to_mutation: Optional[Union[Dict[str, str], pd.Series]] = None,
    prediction_sets: Optional[pd.DataFrame] = None,
    palette: Optional[Mapping[str, str]] = None,
    figsize_per_cellline: float = 0.5,
    show_sporadic: bool = True,
):
    """
    Create stacked barplots of i-MAP scores grouped by true class.
    
    Args:
        predictions: DataFrame with columns CellLines, True, Class_0, Class_1, ..., Class_k
        class_name_map: Mapping from class index to class name (e.g., {0: 'WT', 1: 'FUS'})
        cellline_to_mutation: Optional mapping from cell line names to mutation labels
        prediction_sets: Optional DataFrame with columns CellLines, PredictionSet
        palette: Optional color palette for mutations
        figsize_per_cellline: Width per cell line in figure
        show_sporadic: If True, create separate plots for sporadic samples
        
    Returns:
        List of matplotlib Figure objects (one per true class, optionally one for sporadic)
    """
    palette = cast(Mapping[str, str], PALETTE if palette is None else palette)
    
    # Prepare data
    df = predictions.copy()
    if cellline_to_mutation is not None:
        df = _ensure_mutations_column(df, cellline_to_mutation)
    
    # Merge prediction sets if provided
    if prediction_sets is not None:
        df = pd.merge(df, prediction_sets, on='CellLines', how='left')
    
    # Convert to barplot format
    plot_df = _prepare_barplot_data(df, class_name_map)
    
    # Add prediction sets if available
    if prediction_sets is not None:
        pred_set_map = prediction_sets.set_index('CellLines')['PredictionSet'].to_dict()
        plot_df['PredictionSet'] = plot_df['CellLine'].map(pred_set_map)
    
    figures = []
    
    # Separate sporadic if needed
    if show_sporadic and 'Mutations' in df.columns:
        df_sporadic = df[df['Mutations'] == 'sporadic']
        df_genetic = df[df['Mutations'] != 'sporadic']
        plot_df_sporadic = plot_df[plot_df['CellLine'].isin(df_sporadic['CellLines'])]
        plot_df_genetic = plot_df[plot_df['CellLine'].isin(df_genetic['CellLines'])]
    else:
        df_genetic = df
        plot_df_genetic = plot_df
        df_sporadic = None
        plot_df_sporadic = None
    
    # Create a plot for each true class in genetic samples
    for true_class in plot_df_genetic['TrueClass'].unique():
        class_data = plot_df_genetic[plot_df_genetic['TrueClass'] == true_class].copy()
        
        # Sort celllines by their true class probability (descending)
        cellline_order = (
            class_data.groupby('CellLine')['TrueClassProb']
            .first()
            .sort_values(ascending=False)
            .index.tolist()
        )
        
        # Make CellLine categorical with the sorted order
        class_data['CellLine'] = pd.Categorical(
            class_data['CellLine'],
            categories=cellline_order,
            ordered=True
        )
        
        # Reorder PredictedClass so true class appears first (on bottom of stack)
        class_order = [cls for cls in class_name_map.values() if cls != true_class]
        class_order += [true_class]
        class_data['PredictedClass'] = pd.Categorical(
            class_data['PredictedClass'],
            categories=class_order,
            ordered=True
        )
        
        # Sort the data by PredictedClass to ensure proper stacking order
        class_data = class_data.sort_values('PredictedClass')
        
        # Create figure
        fig_width = max(8, len(cellline_order) * figsize_per_cellline)
        fig, ax = plt.subplots(figsize=(fig_width, 4))
        
        # Create stacked barplot
        for pred_class in class_order:
            subset = class_data[class_data['PredictedClass'] == pred_class]
            bottom = None
            for prev_class in class_order:
                if prev_class == pred_class:
                    break
                prev_subset = class_data[class_data['PredictedClass'] == prev_class]
                if bottom is None:
                    bottom = prev_subset.set_index('CellLine')['Probability']
                else:
                    bottom = bottom.add(
                        prev_subset.set_index('CellLine')['Probability'],
                        fill_value=0
                    )
            
            if bottom is None:
                bottom = np.zeros(len(cellline_order))
            else:
                bottom = [bottom.get(cl, 0) for cl in cellline_order]
            
            heights = [
                subset[subset['CellLine'] == cl]['Probability'].values[0]
                if len(subset[subset['CellLine'] == cl]) > 0 else 0
                for cl in cellline_order
            ]
            
            ax.bar(
                range(len(cellline_order)),
                heights,
                bottom=bottom,
                label=pred_class,
                color=palette.get(pred_class, '#888888'),
                width=0.8
            )
        
        ax.set_xticks(range(len(cellline_order)))
        ax.set_xticklabels(cellline_order, rotation=45, ha='right')
        ax.set_xlabel('')
        ax.set_ylabel('i-MAP scores')
        ax.set_title(f'Predicted scores for {true_class} Cell Lines')
        ax.legend(title='Mutation', loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
        fig.tight_layout()
        
        figures.append(fig)
    
    # Create plot for sporadic samples if present
    if show_sporadic and plot_df_sporadic is not None and len(plot_df_sporadic) > 0:
        sporadic_data = plot_df_sporadic.copy()
        
        # Sort celllines by WT probability
        wt_probs = sporadic_data[sporadic_data['PredictedClass'] == 'WT']
        if len(wt_probs) > 0:
            cellline_order = (
                wt_probs.groupby('CellLine')['Probability']
                .first()
                .sort_values(ascending=False)
                .index.tolist()
            )
        else:
            cellline_order = sporadic_data['CellLine'].unique().tolist()
        
        sporadic_data['CellLine'] = pd.Categorical(
            sporadic_data['CellLine'],
            categories=cellline_order,
            ordered=True
        )
        
        class_order = list(class_name_map.values())
        sporadic_data['PredictedClass'] = pd.Categorical(
            sporadic_data['PredictedClass'],
            categories=class_order,
            ordered=True
        )
        
        sporadic_data = sporadic_data.sort_values('PredictedClass')
        
        # Create figure
        fig_width = max(8, len(cellline_order) * figsize_per_cellline)
        fig, ax = plt.subplots(figsize=(fig_width, 4))
        
        # Create stacked barplot
        for pred_class in class_order:
            subset = sporadic_data[sporadic_data['PredictedClass'] == pred_class]
            bottom = None
            for prev_class in class_order:
                if prev_class == pred_class:
                    break
                prev_subset = sporadic_data[sporadic_data['PredictedClass'] == prev_class]
                if bottom is None:
                    bottom = prev_subset.set_index('CellLine')['Probability']
                else:
                    bottom = bottom.add(
                        prev_subset.set_index('CellLine')['Probability'],
                        fill_value=0
                    )
            
            if bottom is None:
                bottom = np.zeros(len(cellline_order))
            else:
                bottom = [bottom.get(cl, 0) for cl in cellline_order]
            
            heights = [
                subset[subset['CellLine'] == cl]['Probability'].values[0]
                if len(subset[subset['CellLine'] == cl]) > 0 else 0
                for cl in cellline_order
            ]
            
            ax.bar(
                range(len(cellline_order)),
                heights,
                bottom=bottom,
                label=pred_class,
                color=palette.get(pred_class, '#888888'),
                width=0.8
            )
        
        ax.set_xticks(range(len(cellline_order)))
        ax.set_xticklabels(cellline_order, rotation=45, ha='right')
        ax.set_xlabel('')
        ax.set_ylabel('i-MAP scores')
        ax.set_title('Predicted scores for Sporadic Cell Lines')
        ax.legend(title='Mutation', loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
        fig.tight_layout()
        
        figures.append(fig)
    
    return figures


def plot_imap_pca(
    predictions: pd.DataFrame,
    class_name_map: Dict[int, str],
    cellline_to_mutation: Optional[Union[Dict[str, str], pd.Series]] = None,
    conformal_regions: Optional[Dict[Tuple[float, int], pd.DataFrame]] = None,
    sporadic_data: Optional[pd.DataFrame] = None,
    n_components: int = 2,
    random_state: int = 42,
    figsize: Tuple[float, float] = (12, 8),
    title: str = 'PCA of multiclass i-MAP scores',
    palette: Optional[Dict[str, str]] = None,
    show_legend: bool = True,
    alpha_levels: Optional[Dict[float, float]] = None
) -> plt.Figure:
    """
    Create a PCA visualization of i-MAP scores with conformal prediction regions.
    
    Projects i-MAP score vectors into 2D PCA space and overlays conformal prediction
    regions as convex hulls. Optionally includes sporadic/unlabeled samples.
    
    Parameters
    ----------
    predictions : pd.DataFrame
        DataFrame with columns 'CellLines', 'True', 'Class_0', 'Class_1', ..., 'Class_k'
    class_name_map : Dict[int, str]
        Mapping from class indices to class names (e.g., {0: 'WT', 1: 'FUS', 2: 'SOD1'})
    cellline_to_mutation : Dict[str, str] or pd.Series, optional
        Mapping from cell line names to mutation labels for coloring points
    conformal_regions : Dict[Tuple[float, int], pd.DataFrame], optional
        Dictionary mapping (quantile, class_index) to DataFrames containing probability
        vectors that define conformal prediction regions. Output from generate_conformal_regions().
    sporadic_data : pd.DataFrame, optional
        DataFrame with sporadic/unlabeled samples to overlay (same format as predictions)
    n_components : int, default=2
        Number of PCA components (currently only 2 is supported for visualization)
    random_state : int, default=42
        Random state for PCA reproducibility
    figsize : Tuple[float, float], default=(12, 8)
        Figure size in inches
    title : str, default='PCA of multiclass i-MAP scores'
        Plot title
    palette : Dict[str, str], optional
        Color palette mapping mutation names to colors. If None, uses PALETTE.
    show_legend : bool, default=True
        Whether to show the legend
    alpha_levels : Dict[float, float], optional
        Mapping from quantile to alpha transparency level for conformal regions.
        If None, uses {lowest_quantile: 0.3, highest_quantile: 0.6}
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
        
    Examples
    --------
    >>> from maps.utils_update import generate_conformal_regions
    >>> # Generate conformal regions
    >>> regions = generate_conformal_regions(
    ...     predictions, quantiles=[0.75, 0.9], n_classes=3, grid_step=0.05
    ... )
    >>> # Create PCA plot
    >>> fig = plot_imap_pca(
    ...     predictions=predictions,
    ...     class_name_map={0: 'WT', 1: 'FUS', 2: 'SOD1'},
    ...     cellline_to_mutation=cellline_to_mutation,
    ...     conformal_regions=regions
    ... )
    
    Notes
    -----
    - PCA is fitted on the main predictions data
    - Conformal regions and sporadic data are transformed using the same PCA
    - Convex hulls are computed for each conformal region in PCA space
    - Points are colored by mutation label if cellline_to_mutation is provided
    """
    from sklearn.decomposition import PCA
    from scipy.spatial import ConvexHull
    
    if palette is None:
        palette = PALETTE.copy()
    
    # Get class columns
    class_cols = [f'Class_{i}' for i in range(len(class_name_map))]
    
    # Add mutation labels if provided
    df = predictions.copy()
    if cellline_to_mutation is not None:
        df = _ensure_mutations_column(df, cellline_to_mutation)
    
    # Fit PCA on the main data
    prob_data = df[class_cols].values
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_coords = pca.fit_transform(prob_data)
    
    df['PCA1'] = pca_coords[:, 0]
    df['PCA2'] = pca_coords[:, 1]
    
    # Process conformal regions if provided
    hull_data_list = []
    if conformal_regions is not None:
        # Determine quantile labels and alpha levels
        unique_quantiles = sorted(set(qt for qt, _ in conformal_regions.keys()))
        
        if alpha_levels is None:
            if len(unique_quantiles) == 1:
                alpha_levels = {unique_quantiles[0]: 0.5}
            else:
                alpha_levels = {
                    unique_quantiles[0]: 0.3,
                    unique_quantiles[-1]: 0.6
                }
        
        threshold_labels = {
            qt: f'{int(qt * 100)}% confidence region'
            for qt in unique_quantiles
        }
        
        for (qt, class_idx), region_df in conformal_regions.items():
            # Transform region to PCA space
            region_probs = region_df[class_cols].values
            region_pca = pca.transform(region_probs)
            
            # Compute convex hull
            try:
                hull = ConvexHull(region_pca)
                hull_points = region_pca[hull.vertices]
                
                class_name = class_name_map[class_idx]
                
                hull_df = pd.DataFrame({
                    'PCA1': hull_points[:, 0],
                    'PCA2': hull_points[:, 1],
                    'quantile': qt,
                    'class_idx': class_idx,
                    'Mutation': class_name,
                    'Confidence': threshold_labels[qt],
                    'alpha': alpha_levels.get(qt, 0.5),
                    'group': f'{qt:.3f}_class{class_idx}'
                })
                hull_data_list.append(hull_df)
            except Exception as e:
                print(f"Warning: Could not compute hull for quantile={qt}, class={class_idx}: {e}")
    
    # Process sporadic data if provided
    df_sporadic = None
    if sporadic_data is not None:
        df_sporadic = sporadic_data.copy()
        sporadic_probs = df_sporadic[class_cols].values
        sporadic_pca = pca.transform(sporadic_probs)
        df_sporadic['PCA1'] = sporadic_pca[:, 0]
        df_sporadic['PCA2'] = sporadic_pca[:, 1]
        df_sporadic['Mutation'] = 'sporadic'
        
        # Add sporadic to palette if not present
        if 'sporadic' not in palette:
            palette = palette.copy()
            palette['sporadic'] = '#000000'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot conformal region hulls
    if hull_data_list:
        hull_all = pd.concat(hull_data_list, ignore_index=True)
        
        for group_id in hull_all['group'].unique():
            group_data = hull_all[hull_all['group'] == group_id]
            mutation = group_data['Mutation'].iloc[0]
            alpha = group_data['alpha'].iloc[0]
            
            # Close the polygon by adding first point at end
            x_coords = list(group_data['PCA1'].values) + [group_data['PCA1'].iloc[0]]
            y_coords = list(group_data['PCA2'].values) + [group_data['PCA2'].iloc[0]]
            
            ax.fill(
                x_coords,
                y_coords,
                color=palette.get(mutation, '#888888'),
                alpha=alpha,
                edgecolor=None,
                label=None  # We'll handle legend separately
            )
    
    # Plot main data points
    if 'Mutations' in df.columns:
        for mutation in df['Mutations'].dropna().unique():
            subset = df[df['Mutations'] == mutation]
            ax.scatter(
                subset['PCA1'],
                subset['PCA2'],
                color=palette.get(mutation, '#888888'),
                s=100,
                label=mutation,
                edgecolors='white',
                linewidths=0.5,
                zorder=3
            )
    else:
        ax.scatter(
            df['PCA1'],
            df['PCA2'],
            color='#888888',
            s=100,
            edgecolors='white',
            linewidths=0.5,
            zorder=3
        )
    
    # Plot sporadic data
    if df_sporadic is not None:
        ax.scatter(
            df_sporadic['PCA1'],
            df_sporadic['PCA2'],
            color=palette.get('sporadic', '#000000'),
            s=100,
            marker='x',
            linewidths=2,
            label=None,  # Don't add to legend
            zorder=4
        )
    
    # Set labels
    var_ratio = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var_ratio[0]:.1%})', fontsize=14)
    ax.set_ylabel(f'PC2 ({var_ratio[1]:.1%})', fontsize=14)
    ax.set_title(title, fontsize=16)
    
    # Create legend
    if show_legend:
        # Mutation legend
        if 'Mutations' in df.columns:
            mutation_handles = [
                plt.Line2D([0], [0], marker='o', color='w', 
                          markerfacecolor=palette.get(mut, '#888888'),
                          markersize=10, label=mut)
                for mut in df['Mutations'].dropna().unique()
            ]
            legend1 = ax.legend(
                handles=mutation_handles,
                title='',
                loc='upper left',
                bbox_to_anchor=(1.02, 1),
                frameon=True
            )
            ax.add_artist(legend1)
        
        # Confidence region legend
        if hull_data_list:
            unique_confidences = sorted(hull_all['Confidence'].unique())
            conf_handles = [
                mpatches.Patch(
                    facecolor='gray',
                    alpha=alpha_levels.get(float(conf.split('%')[0])/100, 0.5),
                    label=conf
                )
                for conf in unique_confidences
            ]
            ax.legend(
                handles=conf_handles,
                title='',
                loc='upper left',
                bbox_to_anchor=(1.02, 0.6),
                frameon=True
            )
    
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    return fig

# --- QC plots
################################################################################# QC plot functions
################################################################################
def plot_cell_count(screen, facets="Mutations", **kwargs):
    """Plot cell count by cell line, mutational background and optional 
    facetting variable.
    
    Args:
        screen: ScreenBase object with metadata containing cell counts.
        facets: Optional facetting variable to split the bar plots by.
        **kwargs: Additional keyword arguments for seaborn facet grid.
    """
    import polars as pl
    
    def barplot_with_sorted_x(data, **kwargs):
        assert not data.empty
        
        sorted_order = data.groupby("CellLines")["NCells"] \
            .sum() \
            .sort_values(ascending=False) \
            .index
        
        sns.barplot(
            data=data, 
            x="CellLines", 
            y="NCells", 
            order=sorted_order, 
            hue="Mutations",
            palette=PALETTE
        )
   
    groups = set(["CellLines", "Mutations", facets]) 
    
    xplot = screen.metadata \
        .group_by(groups) \
        .agg(pl.col("NCells").mean().alias("NCells")) \
        .to_pandas()
    
    # Create a FacetGrid with independent x-axis categories per face
    g = sns.FacetGrid(
        xplot, 
        col=facets, 
        **kwargs,
    )

    g.map_dataframe(barplot_with_sorted_x)

    # Rotate x-axis labels
    for ax in g.axes.flat:
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_xlabel('') 

    g.figure.subplots_adjust(hspace=0.4)
    return g.figure


def boxplot_cell_counts(screen, **kwargs):
    """ Generates boxplots of cell count by mutational background referenced against overall cell count density.
    
    Args:
        screen: ScreenBase object with metadata containing cell counts.
        **kwargs: Additional keyword arguments for seaborn boxplot.
    """    
    import polars as pl
    xplot = screen.metadata \
        .group_by(["CellLines", "Mutations"]) \
        .agg(pl.col("NCells").mean().alias("NCells")) \
        .to_pandas()
    
        
    fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[1, 1])

    # Get x-axis limits from the data
    x_min = xplot['NCells'].min()
    x_max = xplot['NCells'].max()

    # Plot overall density on top with fill
    sns.kdeplot(
        data=xplot, x='NCells', ax=ax1, color='black', fill=True, alpha=0.7
    )
    
    ax1.set_title('Aggregate Density of Cell Counts')
    ax1.set_xlabel('')
    ax1.set_xlim(x_min, x_max)

    # Create boxplot for mutations
    sns.boxplot(
        data=xplot,
        x='NCells',
        y='Mutations',
        orient='h',
        ax=ax2,
        width=0.5,
        showcaps=False,
        **kwargs
    ) 

    ax2.set_xlabel("Number cells")
    ax2.set_ylabel(None)
    ax2.set_xlim(x_min, x_max)

    return fig


def plot_plate(screen, feature_plot):
    "Generates heatmap of selected feature by plate position"
    import polars as pl
    
    # Merge data with metadata and aggregate features by sell
    df_agg = screen.data.group_by("ID").agg(
        [pl.col(c).mean().alias(c) for c in screen.data.columns if c != "ID"]
    )
     
    xplot = screen.metadata.join(df_agg, on="ID", how="inner") \
        .group_by(["Row", "Column"]) \
        .agg(pl.col(feature_plot).mean().alias(feature_plot)) \
        .to_pandas()

    # Pivot the dataframe to create a matrix for the heatmap
    xplot["Row"] = xplot["Row"].astype(int)
    xplot["Column"] = xplot["Column"].astype(int)

    heatmap_data = xplot \
        .pivot(index="Row", columns="Column", values=feature_plot) \
        .sort_index(axis=0) \
        .sort_index(axis=1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # Create the heatmap
    sns.heatmap(
        heatmap_data, 
        cmap="viridis", 
        annot=True, 
        fmt=".0f", 
        linewidths=0.5,
        ax=ax
    )

    # Customize labels
    ax.set_xlabel("Row")
    ax.set_ylabel("Column")
    ax.set_title(f"Heatmap of {feature_plot} by Row and Column")

    return fig


def plot_pca(pca, components=(1,2), hue="Mutations", group="ID", **kwargs):
    "Plots PCA projection for selected components"
    assert(pca.is_fitted)
    
    if components is None:
        components = pca.params.get("components", (1,2))
    
    meta_plot = pca.screen.metadata.select([group, hue])
    
    # Merge PCA data with metadata
    pca_df = pca.fitted
    pcs = ["column_{i}".format(i=ii - 1) for ii in components]
    pca_df = pca_df.select(pcs + [group])
    pca_df = pca_df.join(pca.screen.metadata, on=group)

    # Plot selected principal components
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    sns.scatterplot(
        x=pca_df[pcs[0]], 
        y=pca_df[pcs[1]], 
        hue=pca_df[hue],
        edgecolor="black",
        ax=ax,
        **kwargs
    )
    
    ax.set_xlabel(f"Principal Component {components[0]}")
    ax.set_ylabel(f"Principal Component {components[1]}")
    ax.set_title("PCA projection")
    return fig