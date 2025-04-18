"""
Screens are the basic container for storing and running an entire analysis pipeline, including: data loading and preprocessing, exploratory figures, and quantitative analyses.

The BaseScreen class is modality independent. Modality-specific Screen classes extend the `ScreenBase` class to handle loading of different data types.
"""
from maps.loaders import OperettaLoader
from maps.processing import *
from maps.analyses import *
from maps.screen_utils import categorize

class ScreenBase():
    "Base class for processing data from a screen"
    def __init__(self, params, loader):
        self.params = params
        self.loader = loader(params)
        self.data = None
        self.metadata = None
    
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
        return eval(fun)(self, **args)
        
    def preprocess(self):
        "Run all data processing steps as specified in the preprocessing key of screen params."
        assert self.data is not None
        assert self.metadata is not None
                
        for f, v in self.params.get("preprocess").items():
            self = self._run(f, v)
            
        self.preprocessed = True
        
    def _eda(self):
        "Run eda modules specified in params"
        pass
    
    def run_analysis(self):
        "Run analysis modules as specified in the analysis key of screen params."
        analyses = {}
        for analysis in self.params.get("analysis"):
            analyses[analysis] = eval(analysis)(self)
            analyses[analysis]._run()
        

class ImageScreen(ScreenBase):
    def __init__(self, params):
        super().__init__(params, OperettaLoader)
        self.data = None
        self.metadata = None
        self.preprocessed = False
         
    def load(self, antibody=None):
        "Load selected antibody data"
        if antibody is None:
           antibody = self.params.get("antibody") 
           
        dfmeta, df = self.loader.load_data(antibody=antibody)
        self.data = df
        self.metadata = dfmeta
        

if __name__ == "__main__":
    import json    
    
    with open("/home/kkumbier/als/scripts/python/params.json", "r") as f:
        params = json.load(f)
 
    screen = ImageScreen(params)
    screen.load()
    screen.preprocess()
    screen.run_analysis()
