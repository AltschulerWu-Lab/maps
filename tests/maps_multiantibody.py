import json

from maps.models import MultiAntibody
from maps.fitters import leave_one_out
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
model_name = list(multi_params["analysis"]["MAP"]["model"].keys())[0]
model = eval(model_name)(**multi_params["analysis"]["MAP"])

# Run leave-one-out fitter
print("Running multi antibody leave-one-out (dataloader)...")
multi_results = leave_one_out(screen, model)
print("Multi antibody results:", multi_results["predicted"])

