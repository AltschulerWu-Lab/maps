import json

from maps.models import BinaryLogistic
from maps.fitters import leave_one_out
from maps.screens import ImageScreen

# Use the provided single antibody param file
single_ab_param_file = "/home/kkumbier/maps/template_analyses/params/maps_singleantibody.json"

# Load params
with open(single_ab_param_file, "r") as f:
    single_params = json.load(f)

# Initialize screen
print("Loading single antibody screen...")
screen = ImageScreen(single_params)
screen.load(antibody="FUS/EEA1")
screen.preprocess()

# Initialize model
model_name = list(single_params["analysis"]["MAP"]["model"].keys())[0]
model = eval(model_name)(**single_params["analysis"]["MAP"])

# Run leave-one-out fitter
print("Running single antibody leave-one-out...")
single_results = leave_one_out(screen, model)
print("Single antibody results:", single_results["predicted"])
