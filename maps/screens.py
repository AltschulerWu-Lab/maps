"""
Screens are the basic container for storing and running an entire analysis pipeline, including: data loading and preprocessing, exploratory figures, and quantitative analyses.

The BaseScreen class is modality independent. Modality-specific Screen classes may be defined to handle loading of different data types.
"""
from maps.loaders import OperettaLoader
from maps.processing import *
from maps.analyses import *
from maps.eda import *
from sklearn.preprocessing import LabelEncoder

import numpy as np


class ScreenBase():
    "Base class for processing data from a screen"
    def __init__(self, params, loader):
        self.params = params
        self.loader = loader(params)
        self.data = None
        self.metadata = None
    
    def get_response(self, encode_categorical=True):    
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
        return self.data
            
    def run(self, fun, args):
        "Generic function runner"
        return eval(fun)(self, **args)
        
    def preprocess(self):
        "Run all steps in params preprocessing"
        assert self.data is not None
        assert self.metadata is not None
                
        for f, v in self.params.get("preprocess").items():
            self = self.run(f, v)
            
        self.preprocessed = True
        
    def eda(self):
        "Run eda modules specified in params"
        pass
    
    def run_analysis(self):
        "Run analysis modules specified in params"
        analyses = {}
        for analysis in self.params.get("analysis"):
            analyses[analysis] = eval(analysis)(self)
            analyses[analysis].run()
        

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
        

def categorize(response):
    if np.issubdtype(response.dtype, np.number):
        return response

    unique_response = np.unique(response)
    ordered = list(sorted(unique_response, key=lambda x: x != "WT"))
    return np.array([ordered.index(r) for r in response])

if __name__ == "__main__":
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
    screen.run_analysis()
