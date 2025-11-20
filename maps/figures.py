from typing import Dict, List, Tuple, Optional, Union, Mapping, cast, Any
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from maps.utils import group_predicted, fit_size_model
from matplotlib import patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as mpl_cm

PALETTE = {
    "WT": "#9A9A9A",
    "FUS": "#B24745" ,
    "C9orf72": "#6A6599",
    "sporadic": "#79AF97",
    "SOD1": "#00A1D5",
    "TDP43": "#DF8F44"
}

def ensure_mutations_column(df: pd.DataFrame, cellline_to_mutation: Union[Dict[str, str], pd.Series]) -> pd.DataFrame:
    """Ensure a 'Mutations' column exists by mapping from CellLines using cellline_to_mutation."""
    if 'Mutations' not in df.columns:
        df = df.copy()
        df['Mutations'] = df['CellLines'].map(cellline_to_mutation)
    return df


def order_cell_lines_by_group(
    predictions: pd.DataFrame,
    cellline_to_mutation: Union[Dict[str, str], pd.Series],
) -> Tuple[List[str], Dict[str, str]]:
    """
    Produce a stable ordering of cell lines by mutation groups for heatmaps/plots.

    - Within each mutation group g (excluding WT/sporadic), order by mean Class_1 for rows where ModelGenetics==g.
    - For WT/sporadic, order by overall mean Class_1 across all ModelGenetics.
    Returns (ordered_celllines, cellline_to_group).
    """
    df = ensure_mutations_column(predictions, cellline_to_mutation)
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
            relevant_cells = df[(df['ModelGenetics'] == mutation) & (df['Mutations'] == mutation)]
            ordered = (
                relevant_cells.groupby('CellLines')['Class_1']
                .mean()
                .reset_index()
                .sort_values('Class_1', ascending=False)['CellLines']
                .tolist()
            )
            missing = [c for c in group['CellLines'] if c not in ordered]
            ordered += missing
        else:
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


def plot_imap_heatmap(
    predictions: pd.DataFrame,
    cellline_to_mutation: Union[Dict[str, str], pd.Series],
    cellline_to_age: Optional[Union[Dict[str, float], pd.Series]] = None,
    row_scale: bool = True,
    antibodies_label: Optional[str] = None,
    palette: Optional[Mapping[str, str]] = None,
    figsize: Tuple[int, int] = (12, 8),
):
    """
    Heatmap of i-MAP scores (Class_1) across CellLines (x) and ModelGenetics (y),
    matching the posthoc_imaps notebook style: continuous low-mid-high gradient,
    mutation and age strips above the heatmap, and clean external legends/colorbars.
    """
    import numpy as np
    from matplotlib.colors import Normalize, LinearSegmentedColormap
    from matplotlib.patches import Patch
    from matplotlib.gridspec import GridSpec

    palette = cast(Mapping[str, str], PALETTE if palette is None else palette)

    df = ensure_mutations_column(predictions, cellline_to_mutation)
    if 'ModelGenetics' not in df.columns:
        df = df.copy()
        df['ModelGenetics'] = 'Model'

    # Order cell lines by groups
    ordered_celllines, cellline_to_group = order_cell_lines_by_group(df, cellline_to_mutation)

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
        plot_df['ModelGenetics'] = plot_df['ModelGenetics'] + f"\n({antibodies_label})"

    plot_df['CellLines'] = pd.Categorical(plot_df['CellLines'], categories=ordered_celllines, ordered=True)

    # Pivot for heatmap
    pivot = plot_df.pivot_table(index='ModelGenetics', columns='CellLines', values=fill_col, aggfunc='mean')

    # Continuous low-mid-high colormap
    cmap = LinearSegmentedColormap.from_list('imap_div', ['#014B8D', '#000000', '#D40C11'], N=256)

    # Layout: 3 rows (mutation strip, age strip, heatmap), share x
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, width_ratios=[1, 0.05], height_ratios=[0.3, 0.3, 10], figure=fig)
    ax_mut = fig.add_subplot(gs[0, 0])
    ax_age = fig.add_subplot(gs[1, 0], sharex=ax_mut)
    ax = fig.add_subplot(gs[2, 0], sharex=ax_mut)
    cbar_ax = fig.add_subplot(gs[2, 1])  # heatmap colorbar

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
    group_centers = [(group_positions[i] + group_positions[i+1] - 1) / 2 for i in range(len(unique_groups))]

    # Mutation strip (top) — no ticks needed
    ax_mut.set_ylim(0, 1)
    ax_mut.set_yticks([])
    ax_mut.set_xticks([])
    for i, cl in enumerate(x_labels):
        color = palette.get(cellline_to_group.get(cl, ''), '#ffffff')
        ax_mut.add_patch(mpatches.Rectangle((i, 0), 1, 1, color=color, clip_on=False))
    # Group labels centered
    for center, gname in zip(group_centers, unique_groups):
        ax_mut.text(center + 0.5, 0.5, gname, ha='center', va='center', fontsize=10, color='black', fontweight='bold')
    # Group separators on both strips and heatmap
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

        # Age colorbar (small) and missing legend, placed outside near the age colorbar
        # Create a tiny axes to the right of the heatmap colorbar
        bbox = cbar_ax.get_position()
        age_cax = fig.add_axes([bbox.x1 + 0.02, bbox.y0 + 0.5*(bbox.height-0.2), 0.02, 0.2])
        sm = mpl_cm.ScalarMappable(cmap=age_cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=age_cax)
        # Missing patch legend — anchor next to age colorbar
        age_box = age_cax.get_position()
        age_legend = fig.legend(
            handles=[Patch(facecolor=(0.7, 0.7, 0.7), label='Missing')],
            title='Age', title_fontsize=11,
            loc='upper left',
            bbox_to_anchor=(age_box.x1 + 0.01, age_box.y0 + 0.15),
            bbox_transform=fig.transFigure,
            frameon=True, fontsize=10,
        )

    # Improve spacing; keep space on right for legends/colorbars
    fig.subplots_adjust(top=0.92, right=0.85, hspace=0.05)
    return fig


def plot_imap_scatter(
    predictions: pd.DataFrame,
    cellline_to_mutation: Union[Dict[str, str], pd.Series],
    palette: Optional[Mapping[str, str]] = None,
    figsize: Tuple[int, int] = (8, 6),
):
    """
    Scatter plot of i-MAP score vs model genetics rows with custom mutation positioning and alpha, matching notebook logic.
    Requires columns: ['CellLines','Class_1'] and optionally 'ModelGenetics'. If missing, uses a single row 'Model'.
    """
    import numpy as np

    palette = cast(Mapping[str, str], PALETTE if palette is None else palette)
    df = ensure_mutations_column(predictions, cellline_to_mutation)
    if 'ModelGenetics' not in df.columns:
        df = df.copy()
        df['ModelGenetics'] = 'Model'

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
    scatter_df['mutation_jitter'] = scatter_df['mutation_position'] + np.random.uniform(-0.05, 0.05, size=len(scatter_df))
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
        # Encode per-point alpha in RGBA colors
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


def plot_grouped_predictions(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    mutations_map: Optional[Union[Dict[str, str], pd.Series]] = None,
    ylab: Optional[str] = None,
    sporadics: bool = True,
    palette: Optional[Mapping[str, str]] = None,
):
    """
    New API: Compute average `y` grouped by (x, hue) from a pandas DataFrame that may not contain
    a 'Mutations' column. Provide `mutations_map` to map CellLines->Mutations when needed.
    Plots a bar chart and reports AUROC when hue == 'Mutations'.
    """
    palette = cast(Mapping[str, str], PALETTE if palette is None else palette)
    dfp = df.copy()
    if hue == 'Mutations' and 'Mutations' not in dfp.columns:
        assert mutations_map is not None, "mutations_map is required when hue='Mutations' and df lacks a Mutations column"
        dfp['Mutations'] = dfp['CellLines'].map(mutations_map)
    if sporadics and hue in dfp.columns:
        dfp = dfp[dfp[hue] != 'sporadic']

    ylab = ylab or y
    grouped = dfp.groupby([x, hue])[y].mean().reset_index()

    # Optional AUROC when hue is Mutations (WT vs non-WT)
    title = None
    if hue == 'Mutations':
        ytrue = (grouped['Mutations'] != 'WT').astype(int)
        fpr, tpr, _ = metrics.roc_curve(ytrue, grouped[y], pos_label=1)
        auroc = metrics.auc(fpr, tpr)
        title = f"AUROC = {auroc:.2f}"

    fig, ax = plt.subplots()
    sns.barplot(data=grouped, x=x, y=y, hue=hue, palette=palette, edgecolor='black', ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(ylab)
    if title:
        ax.set_title(title)
    ax.tick_params(axis='x', labelrotation=90)
    return fig


def plot_grouped_predictions_merged(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    score1: str,
    y: str,
    x: str,
    hue: str,
    mutations_map: Optional[Union[Dict[str, str], pd.Series]] = None,
    ylab: Optional[str] = None,
    sporadics: bool = True,
    palette: Optional[Mapping[str, str]] = None,
):
    """New API: Merge grouped summaries from two DataFrames and plot bars ordered by `score1`.
    Adds Mutations via mapping when needed.
    """
    palette = cast(Mapping[str, str], PALETTE if palette is None else palette)
    def prep(d: pd.DataFrame, value: str) -> pd.DataFrame:
        dp = d.copy()
        if hue == 'Mutations' and 'Mutations' not in dp.columns:
            assert mutations_map is not None, "mutations_map is required when hue='Mutations' and df lacks a Mutations column"
            dp['Mutations'] = dp['CellLines'].map(mutations_map)
        if sporadics and hue in dp.columns:
            dp = dp[dp[hue] != 'sporadic']
        g = dp.groupby([x, hue])[value].mean().reset_index()
        return g

    g1 = prep(df1, score1).rename(columns={score1: '__score1'})
    g2 = prep(df2, y)
    merged = pd.merge(g1, g2, on=[x, hue], how='inner')

    # Order x by score1 within each hue
    order = (
        merged[[x, '__score1']]
        .sort_values('__score1', ascending=False)[x]
        .drop_duplicates()
        .tolist()
    )

    fig, ax = plt.subplots()
    sns.barplot(data=merged, x=x, y=y, hue=hue, order=order, palette=palette, edgecolor='black', ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel(ylab or y)
    ax.tick_params(axis='x', labelrotation=90)
    return fig


def plot_map_adjustment_template(
    df: pd.DataFrame,
    mutations_map: Optional[Union[Dict[str, str], pd.Series]] = None,
    model=None,
    X=None,
    sporadics: bool = False,
    palette: Optional[Mapping[str, str]] = None,
    **kwargs,
):
    """New API: Plot MAP score (Class_1) vs cell count (NCells) with adjustment model.
    Ensures Mutations column via mapping.
    """
    palette = cast(Mapping[str, str], PALETTE if palette is None else palette)
    dp = df.copy()
    if 'Mutations' not in dp.columns:
        assert mutations_map is not None, "mutations_map is required to derive Mutations from CellLines"
        dp['Mutations'] = dp['CellLines'].map(mutations_map)
    if not sporadics:
        dp = dp[dp['Mutations'] != 'sporadic']

    if model is None or X is None:
        model, X = fit_size_model(dp)

    fig, ax = plt.subplots()
    sns.scatterplot(data=dp, x='NCells', y='Class_1', hue='Mutations', palette=palette, ax=ax, **kwargs)

    for mut in dp['Mutations'].dropna().unique():
        Xmut = X[X[mut] == 1]
        preds = model.predict(Xmut)
        ax.plot(Xmut['NCells'], preds, label=f'{mut} fit', color=palette.get(mut, '#333333'))

    ax.legend()
    ax.set_xlabel('Number of Cells')
    ax.set_ylabel('MAP score')
    ax.set_title('Number of cells vs MAP scores')
    return fig

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

   
    
def plot_grouped(df, ylab=None, sporadics=True, **kwargs):
    """DEPRECATED: Use plot_grouped_predictions or plot_grouped_predictions_merged.
    This wrapper delegates to figures_archive for backwards compatibility.
    """
    from .archive.figures_archive import plot_grouped as _legacy_plot_grouped
    return _legacy_plot_grouped(df, ylab=ylab, sporadics=sporadics, **kwargs)
    
     
def plot_grouped2(df1, df2, score1, ylab=None, sporadics=True, **kwargs):
    """DEPRECATED: Use plot_grouped_predictions_merged. Delegates to legacy implementation."""
    from .archive.figures_archive import plot_grouped2 as _legacy_plot_grouped2
    return _legacy_plot_grouped2(df1, df2, score1, ylab=ylab, sporadics=sporadics, **kwargs)

def plot_map_adjustment(df, model=None, X=None, sporadics=False, **kwargs):
    """DEPRECATED: Use plot_map_adjustment_template. Delegates to legacy implementation."""
    from .archive.figures_archive import plot_map_adjustment as _legacy_plot_map_adjustment
    return _legacy_plot_map_adjustment(df, model=model, X=X, sporadics=sporadics, **kwargs)



