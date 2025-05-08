"""
MAP scores are defined relative to a classification model. The `BaseModel` 
class serves as a wrapper for fitting the model, generating predictions, and 
computing feature importances. Specific classification models should be defined 
with `_fit`, `_predict`, and `_get_importance` methods. The `BaseModel` class will make calls to the specific model defined in the params.json file under the `model` key. The value should be a string corresponding to the name of the classification model class (e.g., 'Logistic').
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import List

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from maps.model_utils import balanced_sample
from sklearn.ensemble import RandomForestClassifier

class BaseModel():
    def __init__(self, params):
        self.params = params
        self.model_type = list(params.get("model"))[0]
        self.model = eval(self.model_type)(**self.params)
        
    def fit(self, x, y, id_train):
        return self.model._fit(x, y, id_train)
        
    def predict(self, fitted, x, id_test):
        return self.model._predict(fitted, x, id_test)
    
    def get_importance(self, fitted, x):
        return self.model._get_importance(fitted, x)
    

class Logistic():
    """Logistic regression model for binary classification. Additional kwargs as specified in `sklearn.linear_model.LogisticRegression`."""
    def __init__(self, **kwargs):
        self.params = kwargs

    def _fit(self, x: pl.DataFrame, y: List, id_train: pl.Series):
   
        # Convert Polars DataFrame to NumPy array
        idx = x["ID"].is_in(id_train)
        xtrain = x.filter(idx)
        ytrain = y[0][idx]
        
        # sample balance
        seed = self.params.get("seed", 47)
        xtrain, ytrain = balanced_sample(xtrain, ytrain, random_state=seed)
        xtrain = xtrain.drop("ID")
        
        # scale feature matrix
        scaler = StandardScaler()
        xtrain = scaler.fit_transform(xtrain)
 
        # CV grid search with well-based holdout
        kwargs = self.params.get("model")["BinaryLogistic"]    
        model = LogisticRegression(**kwargs)
        model.fit(xtrain, ytrain)
        return {"model": model, "scaler": scaler}

    def _predict(self, model, x: pl.DataFrame, id_test: pl.Series): 
        # Preprocess the new data
        idx = x["ID"].is_in(id_test)
        xtest = model["scaler"].transform(x.filter(idx).drop("ID"))
        ypred = model["model"].predict_proba(xtest)
        
        # Create a dictionary with ID column
        result_dict = {"ID": x.filter(idx)["ID"]}
    
        # Add a column for each class probability
        for i in range(ypred.shape[1]):
            result_dict[f"Ypred{i}"] = ypred[:, i] 
            
        return pl.DataFrame(result_dict)

    def _get_importance(self, model, x: pl.DataFrame):
        # Extract feature importance from the model
        weights = model["model"].coef_[0]
        features = x.drop("ID").columns
        importance = pd.Series(weights, index=features)
        importance = importance.sort_values(ascending=False)
        
        return importance
    
        
class BinaryLogistic(Logistic):
    """Logistic regression model for binary classification. Additional kwargs as specified in `sklearn.linear_model.LogisticRegression`."""
    def _predict(self, model, x: pl.DataFrame, id_test: pl.Series):
        result_df = super()._predict(model, x, id_test)
        
        # Note: some previous runs had used Y0 instead!
        result_df = result_df.select(["ID", "Ypred1"])
        return result_df.rename({"Ypred2": "Ypred"})

    
class RandomForest():
    """Random forest model for binary classification. Additional kwargs as specified in `sklearn.ensemble.RandomForestClassifier`."""
    def __init__(self, **kwargs):
        self.params = kwargs

    def _fit(self, x: pl.DataFrame, y: List, id_train: pl.Series):
        
        # Convert Polars DataFrame to NumPy array
        idx = x["ID"].is_in(id_train)
        xtrain = x.filter(idx)
        ytrain = y[0][idx]
        
        # sample balance
        seed = self.params.get("seed", 47)
        xtrain, ytrain = balanced_sample(xtrain, ytrain, random_state=seed)
        xtrain = xtrain.drop("ID")
        
        # fit model
        kwargs = self.params.get("model")["RandomForest"]
        model = RandomForestClassifier(**kwargs)
        model.fit(xtrain, ytrain)
        return {"model": model}

    def _predict(self, model, x: pl.DataFrame, id_test: pl.Series):
        # Preprocess and predict
        idx = x["ID"].is_in(id_test)
        xtest = x.filter(idx).drop("ID")
        ypred = model["model"].predict(xtest)
        return pl.DataFrame({"ID": x.filter(idx)["ID"], "Ypred": ypred})
    
    def _get_importance(self, model, x: pl.DataFrame):
        # Extract feature importance from the model
        importance = pd.Series(
            model["model"].feature_importances_,
            index=x.drop("ID").columns
        ).sort_values(ascending=False)
        return importance

        
class BinaryRandomForest(RandomForest):
    """Logistic regression model for binary classification. Additional kwargs as specified in `sklearn.linear_model.LogisticRegression`."""
    def _predict(self, model, x: pl.DataFrame, id_test: pl.Series):
        return super()._predict(model, x, id_test)[:,1]

