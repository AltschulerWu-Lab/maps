import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl

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