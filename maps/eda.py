import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca(screen):
    "Runs PCA on screen data, aggregated by well"
    x_well = screen.data.group_by("ID").mean()
    xid = x_well.select("ID")
    xfeat = x_well.drop("ID")

    # Convert to NumPy array and standardize features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(xfeat.to_numpy())

    # Perform PCA
    pca = PCA()
    principle_components = pca.fit_transform(data_scaled)
    
    pca_df = pl.DataFrame(principle_components) \
        .with_columns(xid)
    
    return pca_df

def plot_pca(pca_df: pl.DataFrame, screen, components=(1, 2), hue="Mutations"):
    "Plots PCA projection"
    pcs = ["column_{i}".format(i=ii - 1) for ii in components]
    pca_df = pca_df.select(pcs + ["ID"])
    pca_df = pca_df.join(screen.metadata, on="ID")

    # Plot the first two principal components
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_df[pcs[0]], y=pca_df[pcs[1]], hue=pca_df[hue])
    plt.xlabel(f"Principal Component {components[0]}")
    plt.ylabel(f"Principal Component {components[1]}")
    plt.title("PCA projection")
    plt.grid(True)

    return plt


def plot_cell_count(screen):
    "Generates plot of average cell count by cell line"
    xplot = screen.metadata \
        .group_by(["CellLines", "Mutations", "Drugs"]) \
        .agg(pl.col("NCells").mean().alias("NCells")) \
        .to_pandas()

    cell_order = xplot \
        .groupby("CellLines")["NCells"] \
        .sum().sort_values(ascending=False) \
        .index \
        .tolist()

    # Create a FacetGrid to make separate barplots for each "Drugs" value
    g = sns.FacetGrid(xplot, col="Drugs", sharey=False, height=5, aspect=1.5)

    g.map_dataframe(
        sns.barplot, 
        x="CellLines", 
        y="NCells", 
        order=cell_order
    )

    # Rotate x-axis labels
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    return plt

def plot_cell_count_plate(screen):
    "Generates heatmap of cell count by well position"
    xplot = screen.metadata \
        .group_by(["CellLines", "Mutations", "Drugs", "Row", "Column"]) \
        .agg(pl.col("NCells").mean().alias("NCells")) \
        .to_pandas()

    # Pivot the dataframe to create a matrix for the heatmap
    xplot["Row"] = xplot["Row"].astype(int)
    xplot["Column"] = xplot["Column"].astype(int)

    heatmap_data = xplot.pivot(index="Row", columns="Column", values="NCells")
    heatmap_data = heatmap_data.sort_index(axis=0).sort_index(axis=1)

    # Create the heatmap
    plt.figure(figsize=(12, 8))

    sns.heatmap(
        heatmap_data, 
        cmap="viridis", 
        annot=True, 
        fmt=".0f", 
        linewidths=0.5
    )

    # Customize labels
    plt.xlabel("Row")
    plt.ylabel("Column")
    plt.title("Heatmap of NCells by Row and Column")

    return plt