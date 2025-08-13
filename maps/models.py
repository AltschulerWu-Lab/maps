"""
MAP scores are defined relative to a classification model. The `BaseModel` 
class serves as a wrapper for fitting the model, generating predictions, and 
computing feature importances. Specific classification models should be defined 
with `_fit`, `_predict`, and `_get_importance` methods. The `BaseModel` class will make calls to the specific model defined in the params.json file under the `model` key. The value should be a string corresponding to the name of the classification model class (e.g., 'Logistic').
"""
import torch
import polars as pl
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from maps.model_utils import balanced_sample
from sklearn.ensemble import RandomForestClassifier

from maps.multiantibody.models import MultiAntibodyClassifier
from maps.multiantibody.training import train
from maps.multiantibody.config import TrainConfig, ModelConfig

class BaseModel():
    def __init__(self, params):
        self.params = params
        self.model_type = list(params.get("model"))[0]
        self.model = eval(self.model_type)(**self.params)
        
    def fit(self, x=None, y=None, id_train=None, data_loader=None):
        """Fit model with either x,y,id_train or data_loader"""
        if data_loader is not None:
            return self.model._fit(data_loader=data_loader)
        elif x is not None and y is not None and id_train is not None:
            return self.model._fit(x=x, y=y, id_train=id_train)
        else:
            raise ValueError(
                "Must provide either (x, y, id_train) or data_loader"
            )
    
    def predict(self, fitted, x=None, id_test=None, data_loader=None):
        """Generate predictions with either x,id_test or data_loader"""
        if data_loader is not None:
            return self.model._predict(fitted, data_loader=data_loader)
        elif x is not None and id_test is not None:
            return self.model._predict(fitted, x=x, id_test=id_test)
        else:
            raise ValueError("Must provide either (x, id_test) or data_loader")
    
    def get_importance(self, fitted, x):
        return self.model._get_importance(fitted, x)
    

# --- Logistic regression classification models ---
class Logistic():
    """Logistic regression model for binary classification. Additional kwargs as specified in `sklearn.linear_model.LogisticRegression`."""
    def __init__(self, **kwargs):
        self.params = kwargs

    def _fit(self, x=None, y=None, id_train=None, data_loader=None):
        # Check that we received the expected arguments for this model
        if data_loader is not None:
            raise ValueError("Logistic model does not support data_loader")
        if x is None or y is None or id_train is None:
            raise ValueError("Logistic model requires x, y, and id_train")
        
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
        xtrain = scaler.fit_transform(xtrain.to_numpy())
        
        # fit model
        model_params = self.params.get("model")
        if model_params is not None and "BinaryLogistic" in model_params:
            kwargs = model_params["BinaryLogistic"]
        else:
            kwargs = {}
        
        model = LogisticRegression(**kwargs)
        model.fit(xtrain, ytrain)
        return {"model": model, "scaler": scaler}

    def _predict(self, model, x=None, id_test=None, data_loader=None):
        # Check that we received the expected arguments for this model
        if data_loader is not None:
            raise ValueError("Logistic model does not support data_loader")
        if x is None or id_test is None:
            raise ValueError("Logistic model requires x and id_test")

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
    def _predict(self, model, x=None, id_test=None, data_loader=None):
        # Check arguments and delegate to parent
        if data_loader is not None:
            raise ValueError("BinaryLogistic model does not support data_loader")
        if x is None or id_test is None:
            raise ValueError("BinaryLogistic model requires x and id_test")
        
        result_df = super()._predict(model, x=x, id_test=id_test)
        result_df = result_df.select(["ID", "Ypred1"])
        return result_df.rename({"Ypred1": "Ypred"})


# --- Random forest classification models --- 
class RandomForest():
    """Random forest model for binary classification. Additional kwargs as specified in `sklearn.ensemble.RandomForestClassifier`."""
    def __init__(self, **kwargs):
        self.params = kwargs

    def _fit(self, x=None, y=None, id_train=None, data_loader=None):
        # Check that we received the expected arguments for this model
        if data_loader is not None:
            raise ValueError("RandomForest model does not support data_loader")
        if x is None or y is None or id_train is None:
            raise ValueError("RandomForest model requires x, y, and id_train")

        # Convert Polars DataFrame to NumPy array
        idx = x["ID"].is_in(id_train)
        xtrain = x.filter(idx)
        ytrain = y[0][idx]

        # sample balance
        seed = self.params.get("seed", 47)
        xtrain, ytrain = balanced_sample(xtrain, ytrain, random_state=seed)
        xtrain = xtrain.drop("ID")

        # fit model
        model_params = self.params.get("model")
        if model_params is not None and "RandomForest" in model_params:
            kwargs = model_params["RandomForest"]
        else:
            kwargs = {}
        
        model = RandomForestClassifier(**kwargs)
        model.fit(xtrain, ytrain)
        return {"model": model}

    def _predict(self, model, x=None, id_test=None, data_loader=None):
        # Check that we received the expected arguments for this model
        if data_loader is not None:
            raise ValueError("RandomForest model does not support data_loader interface")
        if x is None or id_test is None:
            raise ValueError("RandomForest model requires x and id_test arguments")

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
    def _predict(self, model, x=None, id_test=None, data_loader=None):
        # Check arguments and delegate to parent
        if data_loader is not None:
            raise ValueError("BinaryRandomForest model does not support data_loader")
        if x is None or id_test is None:
            raise ValueError("BinaryRandomForest model requires x and id_test")
        return super()._predict(model, x=x, id_test=id_test)


# --- MultiAntibody classification models ---
class MultiAntibody:
    """Wrapper for multimodal PyTorch models to work with existing framework."""
    def __init__(self, **kwargs):
        self.params = kwargs

    def _fit(self, data_loader=None):
        if data_loader is None:
            raise ValueError("MultiAntibody requires a data_loader.")

        # fit model
        model_params = self.params.get("model")
        if model_params is not None and "MultiAntibody" in model_params:
            kwargs = model_params["MultiAntibody"]
        else:
            kwargs = {"train": TrainConfig(), "model": ModelConfig()}

        model = MultiAntibodyClassifier(**kwargs.get("model", ModelConfig()))
        train(model, data_loader, kwargs.get("train", TrainConfig()))
        return {"model": model}

    def _predict(self,  model, data_loader=None):
        if data_loader is None:
            raise ValueError("MultiAntibody requires a data_loader")
        model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in data_loader:
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)
                all_preds.append(probs.cpu().numpy())

        return np.vstack(all_preds) if all_preds else np.array([])

    def _get_importance(self, model, x):
        # Not implemented for deep models
        return None