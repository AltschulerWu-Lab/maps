#!/usr/bin/env python3
"""
Simple test script to verify the validation checks in ImagingDataset.
"""
import polars as pl
import sys
import os

# Add the maps module to path
sys.path.insert(0, '/home/kkumbier/maps')

from maps.multiantibody.data_loaders import ImagingDataset

def test_validation():
    """Test the validation checks."""
    
    # Create sample data
    data = pl.DataFrame({
        "ID": [1, 2, 3, 4],
        "feature1": [1.0, 2.0, 3.0, 4.0],
        "feature2": [2.0, 3.0, 4.0, 5.0]
    })
    
    metadata = pl.DataFrame({
        "ID": [1, 2, 3, 4],
        "CellLines": ["A", "B", "A", "B"],
        "Mutations": ["WT", "MUT", "WT", "MUT"]
    })
    
    print("Testing valid inputs...")
    try:
        dataset = ImagingDataset(
            data=data,
            metadata=metadata,
            antibody="test_antibody",
            response="Mutations",
            grouping="CellLines"
        )
        print("✓ Valid inputs accepted")
    except Exception as e:
        print(f"✗ Unexpected error with valid inputs: {e}")
        return False
    
    print("\nTesting invalid response column...")
    try:
        dataset = ImagingDataset(
            data=data,
            metadata=metadata,
            antibody="test_antibody",
            response="NonExistentColumn",
            grouping="CellLines"
        )
        print("✗ Should have failed with invalid response column")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught invalid response column: {e}")
    
    print("\nTesting invalid response_map key...")
    try:
        dataset = ImagingDataset(
            data=data,
            metadata=metadata,
            antibody="test_antibody",
            response="Mutations",
            grouping="CellLines",
            response_map={"Mutations": {"INVALID_VALUE": 0, "WT": 1}}
        )
        print("✗ Should have failed with invalid response_map key")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught invalid response_map key: {e}")
    
    print("\nTesting invalid response_map column...")
    try:
        dataset = ImagingDataset(
            data=data,
            metadata=metadata,
            antibody="test_antibody",
            response="Mutations",
            grouping="CellLines",
            response_map={"NonExistentColumn": {"WT": 0, "MUT": 1}}
        )
        print("✗ Should have failed with invalid response_map column")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught invalid response_map column: {e}")
    
    print("\nTesting multiple response columns...")
    try:
        dataset = ImagingDataset(
            data=data,
            metadata=metadata,
            antibody="test_antibody",
            response=["Mutations", "CellLines"],
            grouping="CellLines"
        )
        print("✓ Multiple valid response columns accepted")
    except Exception as e:
        print(f"✗ Unexpected error with multiple valid response columns: {e}")
        return False
    
    print("\nTesting multiple response columns with one invalid...")
    try:
        dataset = ImagingDataset(
            data=data,
            metadata=metadata,
            antibody="test_antibody",
            response=["Mutations", "InvalidColumn"],
            grouping="CellLines"
        )
        print("✗ Should have failed with one invalid response column")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught invalid response column in list: {e}")
    
    print("\nAll validation tests passed!")
    return True

if __name__ == "__main__":
    test_validation()
