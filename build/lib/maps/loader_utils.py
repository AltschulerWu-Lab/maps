import re
import os

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
    