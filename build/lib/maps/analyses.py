"""
Analysis classes define specific analysis modules. Parameters for analysis modules should be defined under the `analysis` key of your params.json file—with one entry for each type of analysis to be performed.

Analysis modules output a dictionary of figures and tables—the specific figure and table generated are defined within the analysis module. The `run` method in each analysis module calls all functions necessary to generate the desired output.
"""

from maps.models import *
from maps.fitters import *
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from typing import Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from maps.screens import ScreenBase

class PCA():
    """Principal Component Analysis for dimensionality reduction.
    
    Performs PCA on molecular profiles (imaging features) aggregated by well
    or other grouping. Useful for exploratory data analysis and visualization
    of high-dimensional data.
    
    Attributes:
        - `screen` (ScreenBase): Screen object containing data
        - `fitted` (pl.DataFrame): PCA-transformed data with principal components
        - `is_fitted` (bool): Whether PCA has been fitted
    
    Example:
        ```python
        pca = PCA(screen)
        pca.fit()
        # Access results
        print(pca.fitted)
        ```
    """
    def __init__(self, screen: 'ScreenBase'):
        """Initialize PCA analysis.
        
        Args:
            screen (ScreenBase): Screen object with loaded and preprocessed data.
        """
        self.screen = screen

    def run(self):
        self.fit()
        self.outputs = {"tables": self.fitted}
        
    def fit(self, group: Union[str, List[str]] = "ID"):
        """Run PCA on screen data aggregated by specified grouping.
        
        Features are standardized before PCA. Data is first aggregated by the
        specified grouping variable(s), then PCA is performed on the aggregated
        features.
        
        Args:
            group (str or List[str], optional): Column(s) to group by before PCA.
                Defaults to "ID" (well-level aggregation). Can be "ID" and another
                column for hierarchical grouping.
        """
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
    """Molecular ALS Phenotype (MAP) scoring analysis.
    
    Trains classification models to generate MAP scores (predicted probabilities)
    for distinguishing ALS genetic backgrounds from healthy controls. Scores are
    computed on held-out samples using cross-validation strategies defined by
    fitters.
    
    Attributes:
        - `screen` (ScreenBase): Screen object containing data
        - `params` (dict): MAP analysis parameters from config
        - `model` (BaseModel): Model instance for classification
        - `fitter` (str): Name of fitter function to use
        - `fitted` (Dict): Results dictionary from fitter containing predictions,
            fitted models, and importances
        - `is_fitted` (bool): Whether analysis has been fitted
    
    Example:
        ```python
        # Parameters should define model and fitter in config
        map_analysis = MAP(screen)
        map_analysis.fit()
        # Access predictions
        predictions = map_analysis.fitted["predicted"]
        ```
    """
    def __init__(self, screen: 'ScreenBase'):
        """Initialize MAP analysis from screen parameters.
        
        Args:
            screen (ScreenBase): Screen object with params containing MAP
                configuration under params["analysis"]["MAP"].
                
        Raises:
            AssertionError: If MAP key not found in analysis parameters.
        """
        self.screen = screen
        
        analysis_params = screen.params.get("analysis")
        assert("MAP" in analysis_params.keys())
        self.params = analysis_params.get("MAP")
        
        model = list(self.params.get("model").keys())[0]
        self.model = eval(model)(**self.params)
        
        self.fitter = self.params.get("fitter")
        self.is_fitted = True

    def run(self):
        """Run complete MAP scoring workflow.
        
        Fits models using specified fitter and stores results in outputs
        dictionary.
        """
        self.fit()
        self.outputs = {"tables": self.make_table()}
        
    def fit(self):
        """Fit classification models using specified fitter.
        
        Calls the fitter function specified in params (e.g., leave_one_out,
        sample_split) to train models and generate predictions on held-out data.
        Results are stored in self.fitted.
        """
        self.fitted = eval(self.fitter)(self.screen, self.model)
        self.is_fitted = True

    def make_table(self) -> pl.DataFrame:
        """Generate standard MAP score table.
        
        Returns single-cell predictions with metadata joined.
        
        Returns:
            pl.DataFrame: DataFrame with columns including ID, Ypred (MAP scores),
                and metadata columns (CellLines, Mutations, etc.).
                
        Raises:
            AssertionError: If fit() has not been called.
        """
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
    
