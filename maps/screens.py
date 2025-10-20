"""
Screens are the basic container for storing and running an entire analysis pipeline, including: data loading and preprocessing, exploratory figures, and quantitative analyses.

The BaseScreen class is modality independent. Modality-specific Screen classes extend the `ScreenBase` class to handle loading of different data types.
"""
from maps.loaders import OperettaLoader
from maps.screen_utils import categorize
import polars as pl
import numpy as np
import importlib

class ScreenBase():
    "Base class for processing data from a screen"
    def __init__(self, params, loader):
        self.params = params
        self.loader = loader(params)
        self.data = pl.DataFrame()
        self.metadata = pl.DataFrame()
    
    def get_response(self, encode_categorical=True):
        "Generate vector of response values as specified by analysis.MAP.response key of screen params."    
        
        # Load response vectors as indicated in params
        assert(self.data is not None)
        assert(self.metadata is not None)
        
        assert(
            self.params.get("analysis").get("MAP").get("response") is not None
        )
        
        response = self.params.get("analysis").get("MAP").get("response")
        response = [response] if type(response) is not list else response
    
        y = self.metadata.select(response + ["ID"])
        y = self.data.select("ID").join(y, on="ID").select(response)
        y = [y[col].to_numpy() for col in y.columns]
            
        # Encode y classes as numeric
        if encode_categorical:
            y = [categorize(yy) for yy in y]
            
        return y
    
    def get_data(self):
        "Generate data matrix"
        return self.data
            
    def _run(self, fun, args):
        "Generic function runner"
        return fun(self, **args)
        
    def preprocess(self):
        "Run all data processing steps as specified in the preprocessing key of screen params."
        assert self.data is not None
        assert self.metadata is not None
        processing = importlib.import_module("maps.processing")
        
        for f, v in self.params.get("preprocess").items():
            self = self._run(getattr(processing, f), v)
            
        self.preprocessed = True
        

class ImageScreen(ScreenBase):
    def __init__(self, params):
        super().__init__(params, OperettaLoader)
        self.preprocessed = False
        self.loaded = False
         
    def load(self, antibody=None):
        "Load selected antibody data"
        if antibody is None:
           antibody = self.params.get("antibody") 

        if isinstance(antibody, list):
            raise ValueError("Use `ImageScreenMultiAntibody` to handle multi antibody analysis")
         
        dfmeta, df = self.loader.load_data(antibody=antibody)
        self.data = df
        self.metadata = dfmeta
        self.loaded = True

class ImageScreenMultiAntibody(ScreenBase):
    "Screen class for multimodal data"
    def __init__(self, params):
        super().__init__(params, OperettaLoader)
        self.preprocessed = False
        
    def load(self, antibody=None):
        "Load data for selected antibodies and group by antibody"
        if antibody is None:
            antibody = self.params.get("antibody")
        assert antibody is not None, "Antibody must be specified"

        dfmeta, df = self.loader.load_data(antibody=antibody)

        # Group by antibody
        self.data = {}
        self.metadata = {}
        antibodies = dfmeta["Antibody"].unique()
        
        for ab in antibodies:
            self.metadata[ab] = dfmeta.filter(pl.col("Antibody") == ab)
            self.data[ab] = df.filter(
                pl.col("ID").is_in(self.metadata[ab]["ID"])
            )

    def preprocess(self):
        "Run preprocessing steps for each multimodal dataset"
        assert self.data is not None
        assert self.metadata is not None
        
        df_multimodal = self.data.copy()
        dfmeta_multimodal = self.metadata.copy()
        processing = importlib.import_module("maps.processing")
        
        for ab in df_multimodal:
            self.data, self.metadata = df_multimodal[ab], dfmeta_multimodal[ab]
    
            for f, v in self.params.get("preprocess").items():
                self = self._run(getattr(processing, f), v)

            df_multimodal[ab], dfmeta_multimodal[ab] = self.data, self.metadata
            
        self.data = df_multimodal
        self.metadata = dfmeta_multimodal
        self.preprocessed = True
        
        print("Preprocessing complete")        
        
    def get_response(self, encode_categorical=True):
        "Generate vector of response values as specified by analysis.MAP.response key of screen params."    
        
        # Load response vectors as indicated in params
        assert(self.data is not None)
        assert(self.metadata is not None) 
        
        response = self.params.get("analysis").get("MAP").get("response")
        assert response is not None, "Response must be specified in params"
        response = [response] if type(response) is not list else response
   
        y = {}
        for k in self.data: 
            yy = self.metadata[k].select(response + ["ID"])
            yy = self.data[k].select("ID").join(yy, on="ID").select(response)
            y[k] = [yy[col].to_numpy() for col in yy.columns]
        
        # Check that each response array contains the same set of values
        value_sets = [set(np.concatenate(arr)) for arr in y.values()]
        first_set = value_sets[0]
        for _, s in enumerate(value_sets[1:], start=1):
            assert s == first_set, "Screens contain different genetics"
        
        # Encode y classes as numeric
        if encode_categorical:
            for k in y:
                y[k] = [categorize(yy) for yy in y[k]]
            
        return y

if __name__ == "__main__":
    import json    
    
    pdir = "/home/kkumbier/als/scripts/maps/template_analyses/params/"
    with open(pdir + "params_multimodal.json", "r") as f:
        params = json.load(f)
 
    screen = ImageScreen(params)
    screen.load(antibody="COX IV/Galectin3/atubulin")
    screen.preprocess()
    screen.run_analysis()
    
    mscreen = ImageScreenMultiAntibody(params)
    mscreen.load(antibody=["FUS/EEA1", "COX IV/Galectin3/atubulin"])
    mscreen.preprocess()