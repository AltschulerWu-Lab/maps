"""
Screens are the basic container for storing and running an entire analysis pipeline, including: data loading and preprocessing, exploratory figures, and quantitative analyses.

The BaseScreen class is modality independent. Modality-specific Screen classes extend the `ScreenBase` class to handle loading of different data types.

The module also provides a utility function `merge_screens_to_multiantibody` for combining
multiple single-antibody ImageScreen instances into an ImageScreenMultiAntibody instance.
"""
from maps.loaders import OperettaLoader
from maps.screen_utils import categorize
import polars as pl
import numpy as np
import importlib
from typing import Dict
import copy

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
        self.loaded = False
        
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
            
        self.loaded = True

    def preprocess(self):
        """Run preprocessing steps for each antibody dataset.
        
        Applies all preprocessing functions defined in params to each antibody's
        data independently, maintaining alignment across samples. Handles both
        unified preprocessing (single dict) and antibody-specific preprocessing
        (dict of dicts keyed by antibody).
        """
        assert self.data is not None
        assert self.metadata is not None
        
        df_multimodal = self.data.copy()
        dfmeta_multimodal = self.metadata.copy()
        processing = importlib.import_module("maps.processing")
        
        # Check if preprocess is a dict of dicts (antibody-specific) or single dict
        preprocess_config = self.params.get("preprocess", {})
        is_antibody_specific = False
        
        # Detect if preprocessing is antibody-specific by checking if values are dicts
        if preprocess_config and isinstance(next(iter(preprocess_config.values()), None), dict):
            # Check if keys match antibody names
            if set(preprocess_config.keys()) == set(df_multimodal.keys()):
                is_antibody_specific = True
        
        for ab in df_multimodal:
            self.data, self.metadata = df_multimodal[ab], dfmeta_multimodal[ab]
            
            # Get preprocessing config for this antibody
            if is_antibody_specific:
                ab_preprocess = preprocess_config.get(ab, {})
            else:
                ab_preprocess = preprocess_config
            
            # Apply preprocessing functions
            for f, v in ab_preprocess.items():
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

def merge_screens_to_multiantibody(screens: Dict[str, ImageScreen]) -> ImageScreenMultiAntibody:
    """Merge multiple ImageScreen instances into a single ImageScreenMultiAntibody instance.
    
    This function combines data from multiple single-antibody ImageScreen instances into
    a multi-antibody screen. It validates that analysis parameters are consistent across
    screens and intelligently handles preprocessing parameters that may differ by antibody.
    
    Args:
        screens (Dict[str, ImageScreen]): Dictionary mapping antibody names to ImageScreen
            instances. Keys should be antibody names (e.g., "FUS/EEA1") and values should
            be loaded ImageScreen objects. Screens may or may not be preprocessed.
    
    Returns:
        ImageScreenMultiAntibody: A merged multi-antibody screen with:
            - data: Dict mapping antibody names to DataFrames
            - metadata: Dict mapping antibody names to metadata DataFrames
            - params: Merged parameters with combined antibody list
            - loaded: True (data has been populated from input screens)
            - preprocessed: True if all input screens were preprocessed, False otherwise
    
    Raises:
        ValueError: If screens have inconsistent analysis parameters or if no screens provided.
        AssertionError: If screens are not loaded.
    
    Example:
        >>> screen1 = ImageScreen(params1)
        >>> screen1.load(antibody="FUS/EEA1")
        >>> screen1.preprocess()
        >>> 
        >>> screen2 = ImageScreen(params2)
        >>> screen2.load(antibody="COX IV/Galectin3/atubulin")
        >>> screen2.preprocess()
        >>> 
        >>> screens = {"FUS/EEA1": screen1, "COX IV/Galectin3/atubulin": screen2}
        >>> multi_screen = merge_screens_to_multiantibody(screens)
    """
    if not screens:
        raise ValueError("No screens provided for merging")
    
    # Validate all screens are loaded
    for ab_name, screen in screens.items():
        if not screen.loaded:
            raise AssertionError(f"Screen '{ab_name}' is not loaded")
    
    # Get reference screen for parameter merging
    ref_screen = next(iter(screens.values()))
    
    # Validate and merge params
    merged_params = copy.deepcopy(ref_screen.params)
    
    # Check that analysis parameters are consistent across all screens
    ref_analysis = ref_screen.params.get("analysis", {})
    for ab_name, screen in screens.items():
        screen_analysis = screen.params.get("analysis", {})
        if screen_analysis != ref_analysis:
            raise ValueError(
                f"Analysis parameters for '{ab_name}' differ from reference screen. "
                "All screens must have identical analysis configurations."
            )
    
    # Set antibodies list from screen keys
    merged_params["antibodies"] = list(screens.keys())
    
    # Handle preprocessing parameters
    # Check if all preprocessing configs are identical
    ref_preprocess = ref_screen.params.get("preprocess", {})
    all_same = True
    
    for ab_name, screen in screens.items():
        screen_preprocess = screen.params.get("preprocess", {})
        if screen_preprocess != ref_preprocess:
            all_same = False
            break
    
    if all_same:
        # Use single preprocessing config if all are the same
        merged_params["preprocess"] = ref_preprocess
    else:
        # Create dict of preprocessing configs keyed by antibody
        merged_params["preprocess"] = {
            ab_name: screen.params.get("preprocess", {})
            for ab_name, screen in screens.items()
        }
    
    # Create the multi-antibody screen
    multi_screen = ImageScreenMultiAntibody(merged_params)
    
    # Populate data and metadata from input screens
    multi_screen.data = {}
    multi_screen.metadata = {}
    
    for ab_name, screen in screens.items():
        multi_screen.data[ab_name] = screen.data
        multi_screen.metadata[ab_name] = screen.metadata
    
    # Set loaded flag to True since data and metadata have been populated
    multi_screen.loaded = True
    
    # Set preprocessed flag based on whether all input screens were preprocessed
    multi_screen.preprocessed = all(screen.preprocessed for screen in screens.values())
    
    return multi_screen

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