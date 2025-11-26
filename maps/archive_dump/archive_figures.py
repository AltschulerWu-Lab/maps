import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn import metrics

PALETTE = {
    "WT": "#9A9A9A",
    "FUS": "#B24745" ,
    "C9orf72": "#6A6599",
    "sporadic": "#79AF97",
    "SOD1": "#00A1D5",
    "TDP43": "#DF8F44"
}

def plot_importance(df, groups=["488", "647", "^PC"], group_lab=None, k=25):
    """Computes group-level feature importance—with groups defined by regex matching on columns—and plots group feature importance. """
    importance = np.abs(df.mean(axis=1))
    importance_k = importance.nlargest(k)

    # Count elements matching each group regex
    n_group = [sum(importance.index.str.contains(g)) for g in groups]
    
    importance = pd.Series({
        g: importance_k[importance_k.index.str.contains(g)].sum() 
        for g in groups}
    )
       
    importance = [v / n_group[i] for i, v in enumerate(importance)]
    importance = np.array(importance)
    importance = pd.DataFrame(importance / sum(importance))
    
    if group_lab is None:
        importance.index = groups
    else:
        importance.index = group_lab
 
    df = importance[0].sort_values()
    plt.figure(figsize=(20, 10))
    sns.barplot(y=df.index, x=df.values)
    plt.subplots_adjust(left=0.2)
    plt.ylabel(None)
    plt.xlabel("Importance")
    return plt
    
def adjust_map_scores(df, X, model):
    """ Adjust MAP scores for count"""
    df["Score"] = df["Ypred"] - model.predict(X) \
        + model.params["const"] \
        + model.params["NCells"] * 1000
        
    for mut in df["Mutations"].unique():
       df["Score"] = df["Score"] + (df["Mutations"] == mut) * model.params[mut]
        
    return df


def fit_size_model(df):
    """Fit MAP score cell count adjustment model"""
    X = pd.get_dummies(
        df[['NCells', 'Mutations']], 
        prefix="",
        prefix_sep="",
        dtype=float
    )

    X = sm.add_constant(X)
    model = sm.OLS(df['Ypred'], X).fit()
    return model, X


def plot_map_adjustment(df, model=None, X=None, sporadics=False):
    """Plots MAP score vs cell count with adjustment model"""
    
    if model is None or X is None:
        model, X = fit_size_model(df)
    
    if not sporadics:
        df = df[df["Mutations"] != "sporadic"]
    
    # Plot the data
    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        data=df, x='NCells', y='Ypred', hue='Mutations', palette=PALETTE
    )

    for mut in df["Mutations"].unique():
       
        Xmut = X[X[mut]  == 1]
        predictions = model.predict(Xmut)
        
        # Plot the regression line
        plt.plot(
            Xmut['NCells'], 
            predictions, 
            label=f'{mut} fit', 
            color=PALETTE[mut]
        )

    plt.legend()
    plt.xlabel('Number of Cells')
    plt.ylabel('MAP score')
    plt.title('Number of cells vs MAP scores')
    return plt    
    
    
def plot_count_v_map(df, sporadics=True):
    """Plots MAP scores vs count by genetics, cell line"""
    if not sporadics:
        df = df[df["Mutations"] != "sporadic"]
 
    g = sns.FacetGrid(df, col="Mutations")

    g.map_dataframe(
        sns.scatterplot, x="NCells", y="Ypred", hue="CellLines", legend=False
    )

    g.set_axis_labels("Number of Cells", "MAP score")
    return plt
    
    
def group_predicted(predicted, group, score):
    "Average MAP scores by grouping variable"
    df = predicted \
        .group_by(group) \
        .agg(pl.col(score).mean().alias(score)) \
        .sort(score) \
        .to_pandas() 
        
    return df


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
    _, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, edgecolor="black", ax=ax, **kwargs)

    # Customize plot
    plt.xlabel(None)
    plt.ylabel(ylab)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.2)

    return plt
    
     
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
    _, ax = plt.subplots(figsize=(10, 5))
    
    sns.barplot(
        data=df, 
        edgecolor="black",
        ax=ax,
        **kwargs
    )

    # Customize plot
    plt.xlabel(None)
    plt.ylabel(ylab)
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.2)

    return plt

       
def pca_fig(pca, hue="Mutations"):
    "Plots PCA projection for selected components"
    assert(pca.is_fitted)
    components = pca.params.get("components", (1,2))
    
    "Plots PCA projection"
    pca_df = pca.fitted
    pcs = ["column_{i}".format(i=ii - 1) for ii in components]
    pca_df = pca_df.select(pcs + ["ID"])
    pca_df = pca_df.join(pca.screen.metadata, on="ID")

    # Plot the first two principal components
    plt.figure(figsize=(8, 7))

    sns.scatterplot(
        x=pca_df[pcs[0]], 
        y=pca_df[pcs[1]], 
        hue=pca_df[hue],
        palette=PALETTE,
        edgecolor="black"
    )
    
    plt.xlabel(f"Principal Component {components[0]}")
    plt.ylabel(f"Principal Component {components[1]}")
    plt.title("PCA projection")
    plt.legend(loc='lower left')
    plt.grid(True)
    return plt