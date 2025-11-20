import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from maps.archive.utils import group_predicted, fit_size_model

PALETTE = {
    "WT": "#9A9A9A",
    "FUS": "#B24745" ,
    "C9orf72": "#6A6599",
    "sporadic": "#79AF97",
    "SOD1": "#00A1D5",
    "TDP43": "#DF8F44"
}

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
    """Compute average `score` by plotting group variables (hue, x)""" 
    assert "x" in kwargs.keys()
    assert "y" in kwargs.keys()
    
    if not sporadics:
        df = df.filter(pl.col("Mutations") != "sporadic")
 
    if ylab is None:
        ylab = kwargs["y"]
        
    group = [kwargs["x"], kwargs["hue"]]  
    df = group_predicted(df, group, kwargs["y"])
           
    # Compute AUROC
    if "Mutations" in group:
        ytrue = (df["Mutations"] != "WT").astype(int)
        fpr, tpr, _ = metrics.roc_curve(ytrue, df[kwargs["y"]], pos_label=1)
        auroc = metrics.auc(fpr, tpr)
        title = f"AUROC = {auroc:.2f}"
    else:
        title = None
    
    # Create the barplot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.barplot(data=df, edgecolor="black", ax=ax, **kwargs)

    # Customize plot
    ax.set_xlabel(None)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.tick_params(axis='x', labelrotation=90)
    #ax.subplots_adjust(bottom=0.2)

    return fig
    
     
def plot_grouped2(df1, df2, score1, ylab=None, sporadics=True, **kwargs):
    """Compute and plot average scores from merged dfs. Ordering set based on `score1`. PLotted value is given by `y`, computed for `df2`"""
    assert "x" in kwargs.keys()
    assert "y" in kwargs.keys()
    
    if not sporadics:
        df1 = df1.filter(pl.col("Mutations") != "sporadic")
        df2 = df2.filter(pl.col("Mutations") != "sporadic")
    
    if ylab is None:
        ylab = kwargs["y"]
   
    group = [kwargs["x"], kwargs["hue"]]  
    df1 = group_predicted(df1, group, score1)
    df2 = group_predicted(df2, group, kwargs["y"])
    df = pd.merge(df1, df2, on=group) 
    
    # Create the barplot
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    
    sns.barplot(
        data=df, 
        edgecolor="black",
        ax=ax,
        **kwargs
    )

    # Customize plot
    ax.set_xlabel(None)
    ax.set_ylabel(ylab)
    ax.tick_params(axis='x', labelrotation=90)
    #ax.subplots_adjust(bottom=0.2)

    return fig

def plot_map_adjustment(df, model=None, X=None, sporadics=False, **kwargs):
    """Plots MAP score vs cell count with adjustment model"""
    
    if model is None or X is None:
        model, X = fit_size_model(df)
    
    if not sporadics:
        df = df[df["Mutations"] != "sporadic"]
    
    # Plot the data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    sns.scatterplot(
        data=df, 
        x='NCells', 
        y='Ypred', 
        **kwargs,
        hue='Mutations',
        palette=PALETTE,
        ax=ax
    )

    for mut in df["Mutations"].unique():
       
        Xmut = X[X[mut]  == 1]
        predictions = model.predict(Xmut)
        
        # Plot the regression line
        ax.plot(
            Xmut['NCells'], 
            predictions, 
            label=f'{mut} fit', 
            color=PALETTE[mut]
        )

    ax.legend()
    ax.set_xlabel('Number of Cells')
    ax.set_ylabel('MAP score')
    ax.set_title('Number of cells vs MAP scores')
    return fig 



