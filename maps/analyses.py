"""
Functions below define specific analysis modules. Parameters for analysis modules should be defined under the `analysis` key of your params.json file—with one entry for each type of analysis to be performed.

Analysis modules output a dictionary of figures and tables—the specific figure and table generated are defined within the analysis module. The `run` method in each analysis module calls all functions necessary to generate the desired output.
"""

from maps.models import *
from maps.fitters import *

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class PCAnalysis():
    def __init__(self, screen):
        analysis_params = screen.params.get("analysis")
        assert("PCAnalysis" in analysis_params.keys())
        self.params = analysis_params.get("PCAnalysis") 
        self.screen = screen

    def run(self):
        self.fit()
        self.outputs = {"figs":self.make_fig(), "tables": self.fitted}
        
    def fit(self):
        "Runs PCA on screen data, aggregated by well"
        x_well = self.screen.data.group_by("ID").mean()
        xid = x_well.select("ID")
        xfeat = x_well.drop("ID")

        # Convert to NumPy array and standardize features
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(xfeat.to_numpy())

        # Perform PCA
        pca = PCA()
        principle_components = pca.fit_transform(data_scaled)
        self.fitted = pl.DataFrame(principle_components).with_columns(xid)
        self.is_fitted = True

    def make_fig(self):
        "Plots PCA projection for selected components"
        assert(self.is_fitted)
        components = self.params.get("components", (1,2))
        hue = self.params.get("hue", "Mutations")
        
        "Plots PCA projection"
        pca_df = self.fitted
        pcs = ["column_{i}".format(i=ii - 1) for ii in components]
        pca_df = pca_df.select(pcs + ["ID"])
        pca_df = pca_df.join(self.screen.metadata, on="ID")

        # Plot the first two principal components
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=pca_df[pcs[0]], y=pca_df[pcs[1]], hue=pca_df[hue])
        plt.xlabel(f"Principal Component {components[0] + 1}")
        plt.ylabel(f"Principal Component {components[1] + 1}")
        plt.title("PCA projection")
        plt.grid(True)

        return plt

class MAP():
    def __init__(self, screen):
        analysis_params = screen.params.get("analysis")
        assert("MAP" in analysis_params.keys())
        self.params = analysis_params.get("MAP") 
        self.model = BaseModel(self.params)
        self.fitter = self.params.get("fitter")
        self.screen = screen

    def run(self):
        self.fit()
        self.outputs = {"figs":self.make_fig(), "tables": self.make_table()}
        
    def fit(self):
        self.fitted = eval(self.fitter)(self.screen, self.model)
        self.is_fitted = True

    def make_table(self):
        "Standard MAP score table: MAP scores by single cell w/ cell metadata"
        assert(self.is_fitted) 
        return self.fitted["predicted"]
    
    def make_fig(self):
        "Standard MAP score plot: MAP scores by cell line"
        assert(self.is_fitted)
        
        predicted_avg = self.fitted["predicted"] \
            .group_by(["CellLines", "Mutations"]) \
            .agg(pl.col("Ypred").mean().alias("Ypred")) \
            .with_columns((1 - pl.col("Ypred")).alias("Ypred")) \
            .sort("Ypred")

        # Create the barplot
        df = predicted_avg.to_pandas()
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x="CellLines", y="Ypred", hue="Mutations")

        # Customize plot
        plt.xlabel(None)
        plt.ylabel("MAP score")
        plt.title("MAP scores by Cell Line and Mutations")
        plt.legend(title="Mutations")
        plt.xticks(rotation=90)
        return plt
    
if __name__ == "__main__":
    from screens import ImageScreen
    import os
    import json    
    
    # Set CUDA environment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    
    with open("/home/kkumbier/als/scripts/python/params.json", "r") as f:
        params = json.load(f)
 
    screen = ImageScreen(params)
    screen.load()
    screen.preprocess()
    
    map_analysis = MAP(screen)    
    map_analysis.run()
    
    pca_analysis = PCAnalysis(screen)
    pca_analysis.run()