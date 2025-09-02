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

from maps.loader_utils import get_plate_id

    
class OperettaLoader():
    def __init__(self, params):
        self.params = params
        self._set_project_dir()
        
        if params.get("save_feather", True):
            self._write_feather()
  
    def load_metadata(self, plate_dir):
        "Load in metadata for selected plate"
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

 
    def load_data(self, antibody=None):
        "Wrapper function to load data/metadata and clean antibody names"
        
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
                
        return dfmeta, df


    def list_antibodies(self, plate_ids=None):
        "Wrapper function to load data/metadata and clean antibody names"
        
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
        "Initialize path to screen data directory"
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
   
    def _get_antibody_plates(self, dfmeta, plates, antibody):
        "Filter plates to those containing the specified antibody"
        assert antibody is not None, "Antibody must be specified"
        if type(antibody) is list:
            antibody = "|".join(antibody)

        dfmeta = dfmeta.filter(pl.col("Antibody").str.contains(antibody))
        id_re = "|".join(dfmeta["PlateID"].unique())
        return [p for p in plates if re.search(id_re, p)]
        
    
    def _get_plate_dirs(self):
        """Traverse project directory to find subdirectories for each plate. Return value will be a dict with screen_id as key and plate list as values
        """
        
        plate_dirs = []
        
        for screen in self.project_dir:
            data_dir = os.path.join(screen, "Data") 
            plates = [p for p in glob.glob(f"{data_dir}/**")]
            eval_dir = self.params.get("eval_dir", "Evaluation1")
            plates = [p for p in plates if eval_dir in os.listdir(p)]
            plate_dirs = plate_dirs + plates
            
        return plate_dirs
   
        
    def _get_data_file(self, plate_dir):
        "Set paths to data file for a given plate"
        fname = self.params.get(
            "data_file", "Objects_Population - Nuclei Selected.txt"
        )
       
        eval_dir = self.params.get("eval_dir", "Evaluation1")
        return os.path.join(plate_dir, eval_dir, fname)
            
     
    def _write_feather(self):
        "Write feather files for each plate's data."
        plate_dirs = self._get_plate_dirs()
        eval_dir = self.params.get("eval_dir", "Evaluation1")
        
        for plate_dir in plate_dirs:
            plate_id = get_plate_id(plate_dir)
            out_path = Path(plate_dir) / eval_dir / f"{plate_id}.feather"
            if not os.path.exists(out_path):
                df = self._load_from_csv(plate_dir)
                plate_id = get_plate_id(plate_dir)
                df.write_ipc(out_path)
       
           
    def _clean_data(self, df, plate_dir):
        "Clean and standardize data after loading"
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
    
    def _load_from_csv(self, plate_dir):
        "Load in data for selected plate from csv file"
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

    def _load_from_feather(self, plate_dir):
        "Load in data for selected plate from feather file"
        path = Path(plate_dir) / self.params.get("eval_dir", "Evaluation1")
        path = path / f"{get_plate_id(plate_dir)}.feather"
        
        if not path.exists():
            raise FileNotFoundError(f"Feather file {path} does not exist")
        
        df = pl.read_ipc(path) 
        return df
    
    def _load_data(self, plate_dir):
        "Load in data for selected plate, prioritizing feather files"
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