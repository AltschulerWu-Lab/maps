"""
Analysis classes define specific analysis modules. Parameters for analysis modules should be defined under the `analysis` key of your params.json file—with one entry for each type of analysis to be performed.

Analysis modules output a dictionary of figures and tables—the specific figure and table generated are defined within the analysis module. The `run` method in each analysis module calls all functions necessary to generate the desired output.
"""

from maps.models import *
from maps.fitters import *
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

class PCA():
    def __init__(self, screen):
        self.screen = screen

    def run(self):
        self.fit()
        self.outputs = {"tables": self.fitted}
        
    def fit(self, group="ID"):
        "Runs PCA on screen data, aggregated by well"
        if group == "ID":
            group, drop = [group], None
        else:
            group, drop = ["ID", group], "ID"
        
        x_meta = self.screen.metadata.select(group)
        x_well = x_meta.join(self.screen.data, on="ID", how="inner")
        
        if drop is not None:
            x_well = x_well.drop(drop)
            group = [g for g in group if g != drop]
        
        x_well = x_well.group_by(group).mean()
        xid = x_well.select(group)
        xfeat = x_well.drop(group)

        # Standardize features
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(xfeat.to_numpy())

        # Perform PCA
        pca_model = decomposition.PCA()
        principle_components = pca_model.fit_transform(data_scaled)
        self.fitted = pl.DataFrame(principle_components).with_columns(xid)
        self.is_fitted = True

class MAP():
    def __init__(self, screen):
        analysis_params = screen.params.get("analysis")
        assert("MAP" in analysis_params.keys())
        self.params = analysis_params.get("MAP")
        
        model = list(self.params.get("model").keys())[0]
        self.model = eval(model)(**self.params)
        
        self.fitter = self.params.get("fitter")
        self.screen = screen

    def run(self):
        self.fit()
        self.outputs = {"tables": self.make_table()}
        
    def fit(self):
        self.fitted = eval(self.fitter)(self.screen, self.model)
        self.is_fitted = True

    def make_table(self):
        "Standard MAP score table: MAP scores by single cell w/ cell metadata"
        assert(self.is_fitted) 
        return self.fitted["predicted"]
    
   
if __name__ == "__main__":
    from maps.screens import ImageScreen
    import json    
    
    with open("/home/kkumbier/als/scripts/template_analysis/params/qc.json", "r") as f:
        params = json.load(f)
 
    screen = ImageScreen(params)
    screen.load(antibody="FUS/EEA1")
    screen.preprocess()
    
    pca_analysis = PCA(screen)
    pca_analysis.run()   
    
    map_analysis = MAP(screen)    
    map_analysis.run()
    
