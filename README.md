# `maps`: molecular als phenotype scores
This vignette outlines basic setup and usage for the molecular als phenotype scores (MAPs) pipeline described in Kumbier et al. 2024. Currently, the pipeline supports imaging-based phenotype analysis. However, the pipeline design is fairly modular and can be readily extended to incorporate new data modalities.

## Installation

### Option 1: Using Pre-installed Conda Environment (Recommended)
A pre-configured `maps` conda environment is available on the AW Lab server. **No local conda installation is required** - the commands below use the pre-installed conda at `/awlab/projects/2024_ALS/software/miniforge3/`.

```bash
# Initialize conda (uses the pre-installed version, no local installation needed)
eval "$(/awlab/projects/2024_ALS/software/miniforge3/bin/conda shell.bash hook)"

# Activate the pre-installed maps environment
conda activate maps
```

This will activate the existing `maps` environment without modifying or overwriting the installation. The `eval` command configures your shell to use the pre-installed conda, so you can simply run `conda activate maps` to access the environment.

### Option 2: Install from GitHub with pip
Alternatively, you can install `maps` directly from GitHub using pip. It is recommended to do this within a virtual environment:

```bash
# Create and activate a virtual environment (optional but recommended)
python3.12 -m venv .venv
source .venv/bin/activate

# Install maps from GitHub
pip install git+https://github.com/AltschulerWu-Lab/maps.git
```

### Analysis `params`
To help ensure reproducibility, the `maps` pipeline is organized around parameter (json) files. Parameter files define complete analytical workflows including: which data are used, how data are pre-processed, what quality control analyses are performed, what model is used for classification, and what post-hoc summaries of the model are performed. The `maps` package implements classes for parsing these parameter files and carrying out specified analyses. Any analysis can be explicitly reproduced given its parameter file.

**Analysis metadata params**

- `name`: user-defined name for the analysis
- `description`: brief description of the analysis purpose
- `screen`: screen identifier, specifying the experiment in which the data were generated. Typically, a screen corresponds to cell lines cultured and processed simultaneously
- `antibodies`: list of antibody combinations to include in the analysis (used for multi-antibody analyses)
- `root`: root directory where all screen subdirectories are stored
- `data_file`: name of data file containing molecular profiles (e.g., imaging features)
- `eval_dir`: subdirectory of each plate directory containing data used in the analysis. Typically "Evaluation1" but may be changed if multiple imaging rounds were performed
- `result_dir`: directory where analysis results will be saved

**Preprocessing params**
The `preprocess` dict specifies all preprocessing functions to be performed. Keys indicate the preprocessing function name and values provide kwargs for that function. Common preprocessing functions include:

- `drop_na_features`: Remove features with high proportion of missing values (e.g., `{"na_prop": 0.1}`)
- `drop_sample_by_feature`: Remove specific samples based on metadata (e.g., specific cell lines or mutations)
- `select_sample_by_feature`: Keep only specific samples based on metadata (e.g., only DMSO-treated wells)
- `drop_cells_by_feature_qt`: Remove cells with feature values in extreme quantiles (e.g., outlier cells by size)
- `select_feature_types`: Keep only features matching regex pattern (e.g., intensity or spot features)
- `drop_feature_types`: Remove features matching regex pattern (e.g., sum features or specific channels)
- `drop_constant_features`: Remove features with no variance

For the complete list of preprocessing functions and their parameters, see the [processing documentation](/home/kkumbier/maps/docs/maps/processing.html).

**Analysis params**
The `analysis` dict specifies all analyses to be performed. Currently supports:

- `MAP`: Classification analysis parameters including:
  - `model`: Model specification (e.g., `MultiAntibody` for multi-antibody data)
  - `seed`: Random seed for reproducibility
  - `reps`: Number of replicate runs
  - `data_loader`: Data loading parameters including batch size, number of cells per sample, and response mapping
  - `fitter`: Fitting strategy for train/test splits (e.g., `sample_split_mut` for mutation-based splits)

**Example params file**

```json
{
    "name": "binary", 
    "description": "Binary classification analysis. Models trained to classify a single ALS genetic background vs health. Genetic-specific models applied to all ALS genetic backgrounds in eval set.",
    "screen": "20250216_AWALS37_Full_screen_n96",
    "antibodies": ["FUS/EEA1", "COX IV/Galectin3/atubulin"],
    "root": "/awlab/projects/2024_ALS/Experiments",
    "data_file": "Objects_Population - Nuclei Selected.txt",
    "eval_dir": "Evaluation1",
    "result_dir": "/home/kkumbier/als/analysis_results",
    "preprocess": {
        "drop_na_features": {"na_prop": 0.1},
        "drop_sample_by_feature": {
            "drop_key": [
                {"CellLines": ["C9014", "NS048", "FTD37"]},
                {"Mutations": ["TDP43"]}
            ]
        },
        "select_sample_by_feature": {
            "select_key": [
                {"Drugs": ["DMSO"]}
            ]
        },
        "drop_cells_by_feature_qt": {
            "feature_filters": {
                "Nucleus_Region_Area_[µm²]": 0.05,
                "Cell_Region_Area_[µm²]": 0.05
            }
        },
        "select_feature_types": {
            "feature_str": "^.*Intensity.*$|^.*Spot.*$"
        },
        "drop_feature_types": {
            "feature_str": "^.*Sum.*$|^.*HOECHST.*$|^.*546.*$"
        },
        "drop_constant_features":{}
    },
    "analysis": {
        "MAP": {
            "model": {
                "MultiAntibody": {
                    "model": {
                        "n_classes": 2,
                        "d_model": 16,
                        "n_layers": 1
                    },
                    "train": {
                        "n_epochs": 100
                    }
                }
            },
            "seed": 47,
            "reps": 3,
            "data_loader": {
                "batch_size": 8,
                "n_cells": 250,
                "response_map": {
                    "Mutations": {
                        "WT": 0, 
                        "FUS": 1, 
                        "SOD1": 1, 
                        "C9orf72": 1, 
                        "sporadic": 1
                    }
                } 
            },
            "fitter":  "sample_split_mut"
        }
    }
}
```


### Screens
The `*Screen` classes provide the basic containers for loading and processing data and metadata. Data are loaded based on files/paths defined in `params` by calling `<screen>.load()`. Calling `<screen>.preprocess()` iterates over each preprocessing function defined in `params` and applies that function to the screen data/metadata.

To enable extension to future data modalities, we defined the `ScreenBase` class to handle basic functionality around calling modules from the `maps` package. Modality-specific subclasses (e.g., `ImageScreen`) handle specifics around data I/O that are unique to each modality.

For complete documentation of Screen classes and their methods, see the [screens documentation](/home/kkumbier/maps/docs/maps/screens.html).

#### ImageScreen
Use `ImageScreen` for loading data from a **single antibody combination**. Data are loaded as polars DataFrames with:
- `screen.data`: Single-cell measurements (one row per cell)
- `screen.metadata`: Well-level metadata (one row per imaging well)

Both dataframes share an `ID` column that uniquely identifies each imaging well.

```python
from maps.screens import ImageScreen
from maps.processing import *

# Load params from file
import json
with open("/home/kkumbier/maps/template_analyses/pipelines/params/binary.json", "r") as f:
    params = json.load(f)

# Create screen and load single antibody data
screen = ImageScreen(params)
screen.load(antibody="COX IV/Galectin3/atubulin")

print(screen.data.head())
print(screen.metadata.head())

# Apply preprocessing steps defined in params
screen.preprocess()
```

#### ImageScreenMultiAntibody
Use `ImageScreenMultiAntibody` for loading data from **multiple antibody combinations** simultaneously. This class is designed for multi-modal analyses where data from different antibody stains need to be processed together. Data are loaded as dictionaries of polars DataFrames:
- `screen.data`: Dictionary where keys are antibody names and values are single-cell measurement DataFrames
- `screen.metadata`: Dictionary where keys are antibody names and values are well-level metadata DataFrames

This structure allows preprocessing to be applied independently to each antibody while maintaining alignment across samples.

```python
from maps.screens import ImageScreenMultiAntibody

# Create multi-antibody screen
mscreen = ImageScreenMultiAntibody(params)
mscreen.load(antibody=["FUS/EEA1", "COX IV/Galectin3/atubulin"])

# Access data for each antibody
print("Available antibodies:", list(mscreen.data.keys()))
print("\nFUS/EEA1 data:", mscreen.data["FUS/EEA1"].head())
print("\nCOX IV data:", mscreen.data["COX IV/Galectin3/atubulin"].head())

# Preprocessing is applied to each antibody independently
mscreen.preprocess()
```

**Key differences:**
- `ImageScreen`: Single DataFrame for data and metadata
- `ImageScreenMultiAntibody`: Dictionary of DataFrames (one per antibody)


### Exploratory data analysis (EDA) 
Functions for generating common EDA and quality control figures are defined in `maps.figures`. These can be called directly on `Screen` objects. For example:

```python
from maps.figures import plot_cell_count, PALETTE

fig = plot_cell_count(
    screen, 
    sharex=False, 
    height=4, 
    aspect=1.2
)
fig.show()
```

Quality control plots can help identify outlier samples. For instance, if a cell line shows unusually low cell counts, you can drop it using the `maps.processing.drop_sample_by_feature` function (or specify this in the `params` file preprocessing steps).

PCA is another useful exploratory tool. PCA is performed on molecular profiles (e.g., imaging features) averaged at the level of `ID` (imaging wells):

```python
from maps.analyses import PCA
from maps.figures import plot_pca, PALETTE

pca = PCA(screen)
pca.fit()

fig = plot_pca(pca, palette=PALETTE)
fig.show()
```

### Models and Fitters

The `maps` pipeline provides flexible infrastructure for classification analyses through the `MAP` analysis class. MAP scores are predicted probabilities from classification models trained to differentiate ALS subgroups from healthy controls. Scores are always generated for samples held-out from model training.

#### Models
The specific model used for classification can be one defined in `maps.models` or a new user-defined model. User-defined models should extend the `BaseModel` class and implement:
- `fit_()`: Returns a fitted model
- `predict_()`: Returns model predicted probabilities
- `importance_()`: Returns model feature importances (optional)

Available models include:
- `MultiAntibody`: Multi-modal transformer model for processing multiple antibody stains
- `BinaryLogistic`: Logistic regression for binary classification
- `Delearner`: Two-layer neural network with delearning capabilities

Model parameters (e.g., epochs, batch size, architecture) are specified in the `params` file under `analysis.MAP.model`.

For complete documentation of all available models and their parameters, see the [models documentation](/home/kkumbier/maps/docs/maps/models.html).

#### Fitters
Fitters define how data should be split for training and evaluation. They operate on a `*Screen` object and `BaseModel`, splitting data according to user specifications, training model(s) on the resulting splits, and generating predictions on held-out data.

Available fitters include:
- `sample_split`: Splits samples into train/test sets
- `sample_split_mut`: Mutation-aware splitting for genetic subgroup analyses
- `cross_validation`: K-fold cross-validation

Both models and fitters are specified in the `params` file under the `analysis.MAP` section.

For complete documentation of all available fitters and their parameters, see the [fitters documentation](/home/kkumbier/maps/docs/maps/fitters.html).

### Example Analysis Workflows

Complete analysis workflows demonstrating the full pipeline are available in:
```
/home/kkumbier/maps/template_analyses/pipelines/
```

These notebooks provide examples of:
- Training classification models (`train_models.ipynb`)
- Running complete MAP analyses (`maps_analysis.ipynb`)
- Post-hoc interpretability analyses (`posthoc_imaps.ipynb`, `posthoc_markers.ipynb`)
- Generating analysis notebooks programmatically (`generate_notebooks.py`)

See the `params/` subdirectory for example parameter files used in these workflows.

