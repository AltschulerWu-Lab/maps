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
    """Base class for processing data from a screen.

        This class provides core functionality for loading, preprocessing, and analyzing
        screening data. Modality-specific subclasses should extend this class to handle
        data loading specific to their data type (e.g., imaging data).

        Attributes:

        - params (dict): Analysis parameters including data paths, preprocessing steps,
          and analysis configurations.
        - loader: Data loader instance specific to the data modality.
        - data (pl.DataFrame): Polars DataFrame containing the screening data.
        - metadata (pl.DataFrame): Polars DataFrame containing metadata for the screening data.
        """
    def __init__(self, params, loader):
        """Initialize a ScreenBase instance.

        Args:
            params (dict): Analysis parameters dictionary.
            loader: Loader class to be instantiated for data loading.
        """
        self.params = params
        self.loader = loader(params)
        self.data = pl.DataFrame()
        self.metadata = pl.DataFrame()
    
    def get_response(self, encode_categorical=True):
        """Generate response vector from metadata.

        Extracts response values specified by the ``analysis.MAP.response`` key in
        ``params``.

        Args:
            encode_categorical (bool, optional): Whether to encode categorical variables
                as numeric. Defaults to True.

        Returns:
            list: List of numpy arrays containing response values.
        """
        
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
        """Get the data matrix.

        Returns:
            pl.DataFrame: The screening data.
        """
        return self.data
            
    def _run(self, fun, args):
        """Execute a function with provided arguments.

        Args:
            fun (callable): Function to execute.
            args (dict): Keyword arguments for the function.

        Returns:
            The result of the function call.
        """
        return fun(self, **args)
        
    def preprocess(self):
        """Run all preprocessing steps specified in params.

        Iterates through preprocessing functions defined in ``params['preprocess']``
        and applies them sequentially to the data and metadata.
        """
        assert self.data is not None
        assert self.metadata is not None
        processing = importlib.import_module("maps.processing")
        
        for f, v in self.params.get("preprocess").items():
            self = self._run(getattr(processing, f), v)
            
        self.preprocessed = True
        

class ImageScreen(ScreenBase):
    """Screen class for single-antibody imaging data.

    This class handles loading and processing of imaging data from a single antibody
    combination. Data are loaded as polars DataFrames with single-cell measurements
    in the ``data`` attribute and well-level metadata in the ``metadata`` attribute.

    Attributes:

    - preprocessed (bool): Whether preprocessing has been completed.
    - loaded (bool): Whether data has been loaded.
    - data (pl.DataFrame): Single-cell measurements (one row per cell).
    - metadata (pl.DataFrame): Well-level metadata (one row per imaging well).

    Example:

        >>> screen = ImageScreen(params)
        >>> screen.load(antibody="COX IV/Galectin3/atubulin")
        >>> screen.preprocess()
    """
    def __init__(self, params):
        """Initialize an ImageScreen instance.

        Args:
            params (dict): Analysis parameters dictionary.
        """
        super().__init__(params, OperettaLoader)
        self.preprocessed = False
        self.loaded = False
         
    def load(self, antibody=None):
        """Load data for a single antibody combination.
        
        Args:
            antibody (str, optional): Antibody combination to load. If None, uses
                the 'antibody' key from params. Defaults to None.
        
        Raises:
            ValueError: If a list of antibodies is provided (use ImageScreenMultiAntibody instead).
        """
        if antibody is None:
           antibody = self.params.get("antibody") 

        if isinstance(antibody, list):
            raise ValueError("Use `ImageScreenMultiAntibody` to handle multi antibody analysis")
         
        dfmeta, df = self.loader.load_data(antibody=antibody)
        self.data = df
        self.metadata = dfmeta
        self.loaded = True

class ImageScreenMultiAntibody(ScreenBase):
    """Screen class for multi-antibody imaging data.

    This class handles loading and processing of imaging data from multiple antibody
    combinations simultaneously. Data are organized as dictionaries where keys are
    antibody names and values are polars DataFrames.

    Attributes:

    - preprocessed (bool): Whether preprocessing has been completed.
    - data (dict): Dictionary mapping antibody names to single-cell measurement DataFrames.
    - metadata (dict): Dictionary mapping antibody names to well-level metadata DataFrames.

    Example:

        >>> mscreen = ImageScreenMultiAntibody(params)
        >>> mscreen.load(antibody=["FUS/EEA1", "COX IV/Galectin3/atubulin"])
        >>> mscreen.preprocess()
    """
    def __init__(self, params):
        """Initialize an ImageScreenMultiAntibody instance.
        
        Args:
            params (dict): Analysis parameters dictionary.
        """
        super().__init__(params, OperettaLoader)
        self.preprocessed = False
        
    def load(self, antibody=None):
        """Load data for multiple antibody combinations.
        
        Data are grouped by antibody, with each antibody's data stored separately
        in the data and metadata dictionaries.
        
        Args:
            antibody (list or str, optional): Antibody combinations to load. If None,
                uses the 'antibody' key from params. Defaults to None.
        
        Raises:
            AssertionError: If antibody is None and not specified in params.
        """
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
        """Run preprocessing steps for each antibody dataset.
        
        Applies all preprocessing functions defined in params to each antibody's
        data independently, maintaining alignment across samples.
        """
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
        """Generate response vectors for each antibody from metadata.
        
        Extracts response values specified by the analysis.MAP.response key in params
        for each antibody dataset. Validates that all antibodies have the same set of
        response categories.
        
        Args:
            encode_categorical (bool, optional): Whether to encode categorical variables
                as numeric. Defaults to True.
        
        Returns:
            dict: Dictionary mapping antibody names to lists of numpy arrays containing
                response values.
        
        Raises:
            AssertionError: If response is not specified in params or if different
                antibodies have different response categories.
        """    
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