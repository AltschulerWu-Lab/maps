"""
Test script for MAPs multi-antibody model with leave-one-out fitter (dataloader).
"""
import json

from maps.models import BaseModel
from maps.fitters import leave_one_out_dataloader
from maps.screens import ImageScreenMultiAntibody

# Use the provided multiantibody param file
multi_ab_param_file = "/home/kkumbier/maps/template_analyses/params/maps_multiantibody.json"

# Load params
with open(multi_ab_param_file, "r") as f:
    multi_params = json.load(f)

# Initialize screen
screen = ImageScreenMultiAntibody(multi_params)
screen.load(antibody=["FUS/EEA1", "COX IV/Galectin3/atubulin", "HSP70/SOD1"])
screen.preprocess()

# Initialize model
model = BaseModel(multi_params["analysis"]["MAP"])

# Run leave-one-out fitter
print("Running multi antibody leave-one-out (dataloader)...")
multi_results = leave_one_out_dataloader(screen, model)
print("Multi antibody results:", multi_results["predicted"])

