import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from maps.utils import group_predicted, fit_size_model
from ..figures import PALETTE


def plot_grouped(df, ylab=None, sporadics=True, **kwargs):
    """LEGACY: Compute average `score` by plotting group variables (hue, x). Uses polars.
    Kept for backwards compatibility; prefer new API in figures.py.
    """
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
    ax.set_xlabel('')
    ax.set_ylabel(ylab)
    if title:
        ax.set_title(title)
    ax.tick_params(axis='x', labelrotation=90)
    return fig


def plot_grouped2(df1, df2, score1, ylab=None, sporadics=True, **kwargs):
    """LEGACY: Plot average scores from merged dfs. Uses polars and legacy df format."""
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
    ax.set_xlabel('')
    ax.set_ylabel(ylab)
    ax.tick_params(axis='x', labelrotation=90)
    return fig


def plot_map_adjustment(df, model=None, X=None, sporadics=False, **kwargs):
    """LEGACY: Plots MAP score vs cell count with adjustment model using legacy df format."""
    if model is None or X is None:
        model, X = fit_size_model(df)
    
    if not sporadics:
        df = df[df["Mutations"] != "sporadic"]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    sns.scatterplot(
        data=df, 
        x='NCells', 
        y='Class_1', 
        **kwargs,
        hue='Mutations',
        palette=PALETTE,
        ax=ax
    )

    for mut in df["Mutations"].unique():
        Xmut = X[X[mut]  == 1]
        predictions = model.predict(Xmut)
        ax.plot(
            Xmut['NCells'], 
            predictions, 
            label=f'{mut} fit', 
            color=PALETTE.get(mut, '#333333')
        )

    ax.legend()
    ax.set_xlabel('Number of Cells')
    ax.set_ylabel('MAP score')
    ax.set_title('Number of cells vs MAP scores')
    return fig
