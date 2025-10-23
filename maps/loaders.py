"""
Loaders provide classes for handling data input from different modalities. Every loader must define the methods load_data and load_metadata, which return polars dataframes containing the data and metadata, respectively. Both the `data` and `metdata` should include ID columns that uniquely define each row of metadata. 

**Note** that the `data` dataframe may include multiple rows with the same ID 
value (e.g., when data rows are single cells and metdata correspond to imaging 
wells).

**Note** that loaders will read in the full dataset and write out feather files the first time a screen is processeed. This takes some time, but subsequent runs will be much faster as the data is read from feather files. Setting params `save_feather` = False will skip this step and read data directly from csv files.
"""
import os
import re
import glob
import json
import pandas as pd
import polars as pl   
import numpy as np
from pathlib import Path 
from typing import Optional, List, Union, Tuple

from maps.loader_utils import get_plate_id

    
class OperettaLoader():
    """Loader for Operetta High-Content Imaging System data.
    
    Handles loading of single-cell imaging data and plate metadata from Operetta
    imaging screens. Supports caching via feather files for faster subsequent loads.
    Data is organized by plates, with each plate containing an evaluation directory
    with imaging features and a platemap with well metadata.
    
    Attributes:
        - `params` (dict): Parameter dictionary from json config file
        - `project_dir` (List[Path]): List of paths to screen directories
        - `loader` (OperettaLoader): Loader instance for data I/O
    
    Example:
        ```python
        params = {"root": "/data/Experiments", "screen": "screen_001"}
        loader = OperettaLoader(params)
        metadata, data = loader.load_data(antibody="FUS/EEA1")
        ```
    """
    def __init__(self, params: dict):
        """Initialize OperettaLoader with parameters.
        
        Args:
            params (dict): Configuration dictionary containing:
                - `root` (str): Root directory path
                - `screen` (str or List[str]): Screen identifier(s)
                - `eval_dir` (str, optional): Evaluation subdirectory. Defaults to "Evaluation1"
                - `data_file` (str, optional): Data filename. Defaults to "Objects_Population - Nuclei Selected.txt"
                - `save_feather` (bool, optional): Whether to cache feather files. Defaults to True
        """
        self.params = params
        self._set_project_dir()
        
        if params.get("save_feather", True):
            self._write_feather()
  
    def load_metadata(self, plate_dir: str) -> pl.DataFrame:
        """Load platemap metadata for a selected plate.
        
        Reads platemap CSV file containing well-level metadata (cell lines,
        drugs, antibodies, etc.) and standardizes column names and identifiers.
        
        Args:
            plate_dir (str): Path to plate directory.
            
        Returns:
            pl.DataFrame: Metadata DataFrame with columns including Row, Column,
                Well, CellLines, Drugs, Antibody, Channel, ID, PlateID.
        """
        # Set platemap ath based on data path
        path_parts = Path(plate_dir).parts
        data_idx = path_parts.index("Data") if "Data" in path_parts else -1

        if data_idx != -1:
            parent_path = Path(*path_parts[:data_idx])
            platemap_dir = parent_path / "PlateMap"
    
        else:
            raise Exception("Data directory not found in path")
        
        plate_id = get_plate_id(plate_dir)       
        pm_file = os.path.join(platemap_dir, f"platemap_{plate_id}.csv") 
        df = pl.read_csv(pm_file)
       
        # Standardize column names
        if "Compound" in df.columns:
            df = df.rename({"Compound": "Drugs"})

        if "PlateName" in df.columns:
            df = df.rename({"PlateName": "PlateID"})

        # Set well identifiers
        df = df.with_columns(
            Row=pl.col("Row").map_elements(
                lambda x: str(ord(x) - ord("A") + 1), return_dtype=pl.String
        ))
        
        df = df.with_columns(
            pl.col("Column").cast(pl.String),
            pl.col("PlateID").cast(pl.String),
            pl.col("Channel").cast(pl.String)    
        )
            
        df = df.with_columns(
            ID=pl.col("PlateID") + "-" + pl.col("Row") + "-" + pl.col("Column")
        ) 
        
        return df   
 
    def load_data(self, antibody: Optional[Union[str, List[str]]] = None) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Load data and metadata for selected antibody(ies).
        
        Loads all plates matching the specified antibody pattern, concatenates
        data across plates, and filters to wells with data. Automatically counts
        cells per well and adds screen information.
        
        Args:
            antibody (str or List[str], optional): Antibody name(s) to filter.
                Can be a single antibody string or list of antibodies. Must be
                specified either here or in params.
                
        Returns:
            Tuple[pl.DataFrame, pl.DataFrame]: Tuple of (metadata, data) where:
                - metadata: Well-level metadata with ID, NCells, Screen columns
                - data: Single-cell measurements with ID column for joining
        """
        
        # Load platemaps for selected plates
        plates = self._get_plate_dirs() 
        dfmeta = [self.load_metadata(p) for p in plates]
        dfmeta = pl.concat(dfmeta, how="diagonal_relaxed")
        
        # Filter plates based on selected antibodies
        plates = self._get_antibody_plates(dfmeta, plates, antibody)
        
        # Load data
        df = [self._load_data(p) for p in plates]
        df = pl.concat(df, how="diagonal_relaxed")

        # Count number of cells per well
        df_group = df.group_by("ID").len().rename({"len": "NCells"})
        dfmeta = dfmeta.filter(pl.col("ID").is_in(df_group["ID"]))
        dfmeta = dfmeta.join(df_group, on="ID")

        # Add screen information
        dfmeta = dfmeta.with_columns(
            pl.col("ID").str.split_exact("-", 1).struct.field("field_0").alias("Screen"))
        
        # Only keep columns common to all screens
        ## df case
        df = df.select(
            [col for col in df.columns if df.select(pl.col(col).is_null().sum()).item() < len(df)]
        )
        ## dfmeta case
        dfmeta = dfmeta.select(
            [col for col in dfmeta.columns if dfmeta.select(pl.col(col).is_null().sum()).item() < len(dfmeta)]
        )
        
        return dfmeta, df


    def list_antibodies(self, plate_ids: Optional[List[str]] = None) -> pl.Series:
        """List unique antibodies available in the screen(s).
        
        Args:
            plate_ids (List[str], optional): Plate IDs to filter. If None,
                searches all plates in the screen.
                
        Returns:
            pl.Series: Unique antibody names found in the specified plates.
        """
        
        # Load platemaps for selected plates
        if plate_ids is None:
            plates = self._get_plate_dirs() 
        else:
            id_re = "|".join(plate_ids)
            plates = [p for p in self._get_plate_dirs() if re.search(id_re, p)]
            
        dfmeta = [self.load_metadata(p) for p in plates]
        dfmeta = pl.concat(dfmeta)
        
        return dfmeta["Antibody"].unique()

  
    def _set_project_dir(self):
        """Initialize path to screen data directory.
        
        Validates that screen directories exist and stores them in project_dir.
        Supports loading from multiple screens simultaneously.
        
        Raises:
            FileExistsError: If screen directory does not exist.
        """
        root = self.params.get("root")
        root = "/home/kkumbier/als/data/Experiments" if root is None else root
        
        screen = self.params.get("screen")
        if type(screen) is not list:
            screen = [screen]
        
        project_dir = []
        for s in screen:
                
            screen_dir = Path(root, s)
            
            if not os.path.isdir(screen_dir):
                raise FileExistsError(f"{screen_dir} does not exist")
            
            project_dir.append(screen_dir)
            
        
        self.project_dir = project_dir
   
    def _get_antibody_plates(self, dfmeta: pl.DataFrame, plates: List[str], antibody: Union[str, List[str]]) -> List[str]:
        """Filter plates to those containing the specified antibody.
        
        Args:
            dfmeta (pl.DataFrame): Metadata DataFrame containing Antibody column.
            plates (List[str]): List of plate directory paths.
            antibody (str or List[str]): Antibody name(s) to match.
            
        Returns:
            List[str]: Filtered list of plate directory paths matching antibody.
            
        Raises:
            AssertionError: If antibody is None.
        """
        assert antibody is not None, "Antibody must be specified"
        if type(antibody) is list:
            antibody = "|".join(antibody)

        dfmeta = dfmeta.filter(pl.col("Antibody").str.contains(antibody))
        id_re = "|".join(dfmeta["PlateID"].unique())
        return [p for p in plates if re.search(id_re, p)]
        
    
    def _get_plate_dirs(self) -> List[str]:
        """Traverse project directory to find subdirectories for each plate.
        
        Searches for plate directories containing the specified evaluation
        directory (default: Evaluation1).
        
        Returns:
            List[str]: List of paths to plate directories.
        """
        
        plate_dirs = []
        
        for screen in self.project_dir:
            data_dir = os.path.join(screen, "Data") 
            plates = [p for p in glob.glob(f"{data_dir}/**")]
            eval_dir = self.params.get("eval_dir", "Evaluation1")
            plates = [p for p in plates if eval_dir in os.listdir(p)]
            plate_dirs = plate_dirs + plates
            
        return plate_dirs
   
        
    def _get_data_file(self, plate_dir: str) -> str:
        """Get path to data file for a given plate.
        
        Args:
            plate_dir (str): Path to plate directory.
            
        Returns:
            str: Full path to data file.
        """
        fname = self.params.get(
            "data_file", "Objects_Population - Nuclei Selected.txt"
        )
       
        eval_dir = self.params.get("eval_dir", "Evaluation1")
        return os.path.join(plate_dir, eval_dir, fname)
            
     
    def _write_feather(self):
        """Write feather files for each plate's data.
        
        Creates cached feather files for faster subsequent loads. Only creates
        files that don't already exist. This is called automatically during
        initialization if save_feather=True in params.
        """
        plate_dirs = self._get_plate_dirs()
        eval_dir = self.params.get("eval_dir", "Evaluation1")
        
        for plate_dir in plate_dirs:
            plate_id = get_plate_id(plate_dir)
            out_path = Path(plate_dir) / eval_dir / f"{plate_id}.feather"
            if not os.path.exists(out_path):
                df = self._load_from_csv(plate_dir)
                plate_id = get_plate_id(plate_dir)
                df.write_ipc(out_path)
       
           
    def _clean_data(self, df: pl.DataFrame, plate_dir: str) -> pl.DataFrame:
        """Clean and standardize data after loading.
        
        Filters to imaging features and metadata columns, standardizes column
        names, creates well identifiers (ID), and drops unnecessary columns.
        
        Args:
            df (pl.DataFrame): Raw data DataFrame.
            plate_dir (str): Path to plate directory.
            
        Returns:
            pl.DataFrame: Cleaned DataFrame with ID column and standardized names.
            
        Raises:
            ValueError: If no imaging feature columns are found.
        """
        # Filter to imaging features & row, col metadata, clean column names
        available_cols = df.columns
        pattern_cols = [col for col in available_cols if 
                       re.match(r'^Nuclei Selected.*$|^Row$|^Column$', col)]
        
        if pattern_cols:
            df = df.select(pattern_cols)
        else:
            raise ValueError(
                "No columns matching 'Nuclei Selected', 'Row', or 'Column' found in data"
            )
            
        # Clean column names
        df = df.rename({col: col.replace(" ", "_") for col in df.columns}) 
        
        df = df.rename(
            {col: col.replace("Nuclei_Selected_-_", "") for col in df.columns}
        )
        
        # Set well identifiers
        df = df.with_columns(PlateID=pl.lit(get_plate_id(plate_dir)))
        
        df = df.with_columns(
            ID=pl.col("PlateID") + "-" + pl.col('Row') + "-" + pl.col('Column')
        ) 
       
        # Drop unnecessary columns if they exist
        cols = ["Object_No_in_Nuclei", "Row", "Column", "PlateID"] 
        cols_to_drop = [col for col in cols if col in df.columns]
        
        if cols_to_drop:
            df = df.drop(cols_to_drop)
            
        return df
    
    def _load_from_csv(self, plate_dir: str) -> pl.DataFrame:
        """Load data for selected plate from CSV file.
        
        Parses Operetta CSV format, skipping header lines and setting appropriate
        schema for data types.
        
        Args:
            plate_dir (str): Path to plate directory.
            
        Returns:
            pl.DataFrame: Cleaned data DataFrame.
        """
        data_file = self._get_data_file(plate_dir)
        
        # Get starting line of data
        with open(data_file, 'r') as file:
            for line_number, line in enumerate(file):
                if line.startswith('[Data]'):
                    skip_lines = line_number + 1
                    break

        # Set schema
        df = pd.read_csv(data_file, skiprows=skip_lines, nrows=1, sep="\t")
        schema = {}
        
        for col in df.columns:
            if col in ["Bounding Box", "Row", "Column"]:
                schema[col] = pl.String
            else:
                schema[col] = pl.Float32
        
        df = pl.read_csv(
            source=data_file, 
            skip_rows=skip_lines, 
            separator="\t", 
            null_values = "NaN",
            schema=schema
        )
        
        # Clean and standardize the data
        df = self._clean_data(df, plate_dir)
        return df

    def _load_from_feather(self, plate_dir: str) -> pl.DataFrame:
        """Load data for selected plate from feather file.
        
        Args:
            plate_dir (str): Path to plate directory.
            
        Returns:
            pl.DataFrame: Data DataFrame.
            
        Raises:
            FileNotFoundError: If feather file does not exist.
        """
        path = Path(plate_dir) / self.params.get("eval_dir", "Evaluation1")
        path = path / f"{get_plate_id(plate_dir)}.feather"
        
        if not path.exists():
            raise FileNotFoundError(f"Feather file {path} does not exist")
        
        df = pl.read_ipc(path) 
        return df
    
    def _load_data(self, plate_dir: str) -> pl.DataFrame:
        """Load data for selected plate, prioritizing feather files.
        
        Attempts to load from cached feather file first, falling back to CSV
        if feather file doesn't exist.
        
        Args:
            plate_dir (str): Path to plate directory.
            
        Returns:
            pl.DataFrame: Data DataFrame.
        """
        path = Path(plate_dir) / self.params.get("eval_dir", "Evaluation1")
        path = path / f"{get_plate_id(plate_dir)}.feather"
        
        if os.path.exists(path):
            df = self._load_from_feather(plate_dir)
        else:
            df = self._load_from_csv(plate_dir)
        
        return df
       
  
if __name__ == "__main__":

    pdir = "/home/kkumbier/als/scripts/maps/template_analyses/params/"
    with open(pdir + "params.json", "r") as f:
        params = json.load(f)
        
    self = OperettaLoader(params)
    dfmeta, df = self.load_data(antibody="FUS/EEA1")
    print('Data loaded successfully')