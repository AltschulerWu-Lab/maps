import os
import re
import glob
import json
import pandas as pd
import polars as pl    

def clean_marker_names(df, dfmeta):
    "Replace marker IDs in data columns with antibody names"
    antibodies = dfmeta["Antibody"].str.split("/").explode().unique()
    channels = dfmeta["Channel"].str.split("/").explode().unique()

    for channel, antibody in zip(channels, antibodies):
        df.columns = df.columns.str.replace(channel, antibody)
        
    return df


def get_plate_id(plate_dir):
    "Extract plate ID from plate directory name"
    return re.sub(r"__.*$", "", os.path.basename(plate_dir))
    
    
class OperettaLoader():
    def __init__(self, params):
        self.params = params
        self.__set_project_dir__()
        self.data_dir = os.path.join(self.project_dir, "Data")
        self.platemap_dir = os.path.join(self.project_dir, "PlateMap")  
   
    def __set_project_dir__(self):
        "Initialize path to screen data directory"
        root = self.params.get("root")
        root = "/home/kkumbier/als" if root is None else root
        screen = self.params.get("screen")    
        project_dir = os.path.join(root, "data", "Experiments", screen)

        if not os.path.isdir(project_dir):
            raise FileExistsError
        
        self.project_dir = project_dir
   
    
    def get_plate_dirs(self):
        "Traverse project directory to find subdirectories for each plate"
        plates = [p for p in glob.glob(f"{self.data_dir}/**")]
        return [p for p in plates if "Evaluation1" in os.listdir(p)]
   
        
    def get_data_file(self, plate_dir):
        "Set paths to data file for a given plate"
        fname = self.params.get(
            "data_file", "Objects_Population - Nuclei Selected.txt"
        )
       
        eval_dir = self.params.get("eval_dir", "Evaluation1")
        return os.path.join(plate_dir, eval_dir, fname)
            
       
    def load_metadata(self, plate_dir):
        "Load in metadata for selected plate"
        plate_id = get_plate_id(plate_dir)        
        pm_file = os.path.join(self.platemap_dir, f"platemap_{plate_id}.csv") 
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
            pl.col("PlateID").cast(pl.String)    
        )
            
        df = df.with_columns(
            ID=pl.col("PlateID") + "-" + pl.col("Row") + "-" + pl.col("Column")
        ) 
        
        return df   
   
    
    def load_data_(self, plate_dir):
        "Load in data for selected plate"
        data_file = self.get_data_file(plate_dir)
        
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
        #df = df.rename({"Object_No_in_Nuclei": "CellID"}) 
            
        return df


    def load_data(self, plate_ids=None, antibody=None):
        "Wrapper function to load data/metadata and clean antibody names"
        
        # Load platemaps for selected plates
        if plate_ids is None:
            plates = self.get_plate_dirs() 
        else:
            id_re = "|".join(plate_ids)
            plates = [p for p in self.get_plate_dirs() if re.search(id_re, p)]
            
        dfmeta = [self.load_metadata(p) for p in plates]
        dfmeta = pl.concat(dfmeta)
        
        # Filter to selected antibodies
        antibody = antibody if type(antibody) is list else [antibody]
        
        if antibody is not None:
            ab_re = "|".join(antibody)
            dfmeta = dfmeta.filter(pl.col("Antibody").str.contains(ab_re))
            
        id_re = "|".join(dfmeta["PlateID"].unique())
        plates = [p for p in plates if re.search(id_re, p)]
        
        # Load data
        df = [self.load_data_(p) for p in plates]
        df = pl.concat(df)

        # Count number of cells per well
        df_group = df.group_by("ID").len().rename({"len": "NCells"})
        dfmeta = dfmeta.join(df_group, on="ID")
        
        return dfmeta, df

    def list_antibodies(self, plate_ids=None):
        "Wrapper function to load data/metadata and clean antibody names"
        
        # Load platemaps for selected plates
        if plate_ids is None:
            plates = self.get_plate_dirs() 
        else:
            id_re = "|".join(plate_ids)
            plates = [p for p in self.get_plate_dirs() if re.search(id_re, p)]
            
        dfmeta = [self.load_metadata(p) for p in plates]
        dfmeta = pl.concat(dfmeta)
        
        return dfmeta["Antibody"].unique()
        

if __name__ == "__main__":

    with open("/home/kkumbier/als/scripts/python/params.json", "r") as f:
        params = json.load(f)
        
    self = OperettaLoader(params)
    dfmeta, df = self.load_data(antibody="FUS/EEA1")