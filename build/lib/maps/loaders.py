"""
Loaders provide classes for handling data input from different modalities. Every loader must define the methods load_data and load_metadata, which return polars dataframes containing the data and metadata, respectively. Both the `data` and `metdata` should include ID columns that uniquely define each row of metadata. Note that the `data` dataframe may include multiple rows with the same ID value (e.g., when data rows are single cells and metdata correspond to imaging wells).
"""
import os
import re
import glob
import json
import pandas as pd
import polars as pl   
from pathlib import Path 

from maps.loader_utils import get_plate_id

    
class OperettaLoader():
    def __init__(self, params):
        self.params = params
        self.__set_project_dir__()
        #self.data_dir = os.path.join(self.project_dir, "Data")
        #self.platemap_dir = os.path.join(self.project_dir, "PlateMap")  
   
    def __set_project_dir__(self):
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
   
    
    def _load_data(self, plate_dir):
        "Load in data for selected plate"
        data_file = self._get_data_file(plate_dir)
        
        # Get starting line of data
        with open(data_file, 'r') as file:
            for line_number, line in enumerate(file):
                if line.startswith('[Data]'):
                    skip_lines = line_number + 1
                    break

        # Set schema
        df = pd.read_csv(data_file, skiprows=skip_lines, nrows=1, sep="\t")
        schema = {col:pl.Float32 for col in df.columns}
        schema["Bounding Box"] = pl.String
        schema["Row"] = pl.String
        schema["Column"] = pl.String
        
        df = pl.read_csv(
            source=data_file, 
            skip_rows=skip_lines, 
            separator="\t", 
            null_values = "NaN",
            schema=schema
        )
        
        # Filter to imaging features & row, col metadata, clean column names
        df = df.select(pl.col('^Nuclei Selected.*$|^Row$|^Column$'))
        df = df.rename({col: col.replace(" ", "_") for col in df.columns}) 
        
        df = df.rename(
            {col: col.replace("Nuclei_Selected_-_", "") for col in df.columns}
        )
        
        # Set well identifiers
        df = df.with_columns(PlateID=pl.lit(get_plate_id(plate_dir) ))
        
        df = df.with_columns(
            ID=pl.col("PlateID") + "-" + pl.col('Row') + "-" + pl.col('Column')
        ) 
       
        df = df.drop(["Object_No_in_Nuclei", "Row", "Column", "PlateID"])
        return df


    def load_data(self, antibody=None):
        "Wrapper function to load data/metadata and clean antibody names"
        
        # Load platemaps for selected plates
        plates = self._get_plate_dirs() 
        dfmeta = [self.load_metadata(p) for p in plates]
        dfmeta = pl.concat(dfmeta, how="diagonal_relaxed")
        
        # Filter to selected antibodies
        antibody = antibody if type(antibody) is list else [antibody]
        
        if antibody is not None:
            ab_re = "|".join(antibody)
            dfmeta = dfmeta.filter(pl.col("Antibody").str.contains(ab_re))
            
        id_re = "|".join(dfmeta["PlateID"].unique())
        plates = [p for p in plates if re.search(id_re, p)]
        
        # Load data
        df = [self._load_data(p) for p in plates]
        df = pl.concat(df, how="diagonal_relaxed")

        # Count number of cells per well
        df_group = df.group_by("ID").len().rename({"len": "NCells"})
        dfmeta = dfmeta.join(df_group, on="ID")
        
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
        

if __name__ == "__main__":

    pdir = "/home/kkumbier/als/scripts/maps/template_analyses/params/"
    with open(pdir + "params.json", "r") as f:
        params = json.load(f)
        
    self = OperettaLoader(params)
    dfmeta, df = self.load_data(antibody="FUS/EEA1")