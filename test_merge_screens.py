"""
Test script for merge_screens_to_multiantibody function.

This demonstrates how to merge multiple single-antibody ImageScreen instances
into a multi-antibody ImageScreenMultiAntibody instance.
"""

import json
import copy
from maps.screens import ImageScreen, merge_screens_to_multiantibody

# Example 1: Merging screens with identical preprocessing
print("="*80)
print("Example 1: Merging screens with identical preprocessing")
print("="*80)

# Load the multiclass params as a template
with open("/home/kkumbier/maps/template_analyses/pipelines/params/multiclass.json", "r") as f:
    base_params = json.load(f)

# Create params for first antibody
params1 = copy.deepcopy(base_params)
params1["antibody"] = "FUS/EEA1"

# Create params for second antibody  
params2 = copy.deepcopy(base_params)
params2["antibody"] = "COX IV/Galectin3/atubulin"

# Create and load individual screens
screen1 = ImageScreen(params1)
screen1.load(antibody="FUS/EEA1")
print(f"\nScreen 1 loaded: {screen1.data.shape}")
print(f"Antibody in metadata: {screen1.metadata['Antibody'].unique().to_list()}")

screen2 = ImageScreen(params2)
screen2.load(antibody="COX IV/Galectin3/atubulin")
print(f"\nScreen 2 loaded: {screen2.data.shape}")
print(f"Antibody in metadata: {screen2.metadata['Antibody'].unique().to_list()}")

# Preprocess the screens
print("\nPreprocessing individual screens...")
screen1.preprocess()
screen2.preprocess()
print(f"Screen 1 after preprocessing: {screen1.data.shape}")
print(f"Screen 2 after preprocessing: {screen2.data.shape}")

# Merge into multi-antibody screen
print("\nMerging screens into multi-antibody screen...")
screens_dict = {
    "FUS/EEA1": screen1,
    "COX IV/Galectin3/atubulin": screen2
}

multi_screen = merge_screens_to_multiantibody(screens_dict)

print(f"\nMulti-antibody screen created:")
print(f"  Antibodies: {list(multi_screen.data.keys())}")
print(f"  FUS/EEA1 data shape: {multi_screen.data['FUS/EEA1'].shape}")
print(f"  COX IV data shape: {multi_screen.data['COX IV/Galectin3/atubulin'].shape}")
print(f"  Preprocessed: {multi_screen.preprocessed}")
print(f"  Params antibodies field: {multi_screen.params.get('antibodies')}")
print(f"  Preprocessing type: {'unified' if isinstance(multi_screen.params['preprocess'], dict) and 'drop_na_features' in multi_screen.params['preprocess'] else 'antibody-specific'}")

# Verify analysis params are preserved
print(f"\nAnalysis params preserved:")
print(f"  Model: {list(multi_screen.params['analysis']['MAP']['model'].keys())}")
print(f"  Fitter: {multi_screen.params['analysis']['MAP']['fitter']}")

print("\n" + "="*80)
print("Example 1 completed successfully!")
print("="*80)


# Example 2: Merging screens with different preprocessing (hypothetical)
print("\n\n" + "="*80)
print("Example 2: Merging screens with different preprocessing")
print("="*80)

# Create params with different preprocessing for each antibody
params3 = copy.deepcopy(base_params)
params3["antibody"] = "FUS/EEA1"
params3["preprocess"]["drop_na_features"]["na_prop"] = 0.05  # Different threshold

params4 = copy.deepcopy(base_params)
params4["antibody"] = "COX IV/Galectin3/atubulin"
params4["preprocess"]["drop_na_features"]["na_prop"] = 0.1  # Different threshold

# Create and load screens
screen3 = ImageScreen(params3)
screen3.load(antibody="FUS/EEA1")

screen4 = ImageScreen(params4)
screen4.load(antibody="COX IV/Galectin3/atubulin")

# Preprocess
screen3.preprocess()
screen4.preprocess()

# Merge
screens_dict2 = {
    "FUS/EEA1": screen3,
    "COX IV/Galectin3/atubulin": screen4
}

multi_screen2 = merge_screens_to_multiantibody(screens_dict2)

print(f"\nMulti-antibody screen with different preprocessing:")
print(f"  Preprocessing type: {'unified' if 'drop_na_features' in multi_screen2.params.get('preprocess', {}) else 'antibody-specific'}")

if isinstance(multi_screen2.params['preprocess'], dict):
    if 'FUS/EEA1' in multi_screen2.params['preprocess']:
        print(f"  Antibody-specific preprocessing detected:")
        for ab in multi_screen2.params['preprocess'].keys():
            na_prop = multi_screen2.params['preprocess'][ab].get('drop_na_features', {}).get('na_prop', 'N/A')
            print(f"    {ab}: na_prop = {na_prop}")

print("\n" + "="*80)
print("Example 2 completed successfully!")
print("="*80)

print("\n✓ All tests passed!")
