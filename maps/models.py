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
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model = None
        self.fitted = False
    
    def fit(self, x=None, y=None, id_train=None, data_loader=None):
        pass
    
    def predict(self, x=None, y=None, id_test=None, data_loader=None):
        return pl.DataFrame()
    
    def get_importance(self, x=None):
        pass    
        
class SKLearnModel(BaseModel):
    """Base class for all scikit-learn models."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)        
        self.model_type = "sklearn"

class PyTorchModel(BaseModel):
    """Base class for all PyTorch models."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params = kwargs
        self.model_type = "pytorch"

# --- Logistic regression classification models ---
class Logistic(SKLearnModel):
    """Logistic regression model for binary classification. Additional kwargs as specified in `sklearn.linear_model.LogisticRegression`."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize model and scaler
        model_params = self.params.get("model")
        if model_params is not None and "BinaryLogistic" in model_params:
            kwargs = model_params["BinaryLogistic"]
        elif model_params is not None and "Logistic" in model_params:
            kwargs = model_params["Logistic"]
        else:
            kwargs = {}
        
        self.scaler = StandardScaler()
        self.model = LogisticRegression(**kwargs)
        self.fitted = False
        self.scaled = False
    
    def fit(self, x, y, id_train=None):
        
        # Filter to training ids if provided
        if id_train is not None:
            idx = x["ID"].is_in(id_train)
            xtrain = x.filter(idx)
            ytrain = y[0][idx]
        else:
            xtrain = x
            ytrain = y[0]   
        
        # sample balance
        seed = self.params.get("seed", 47)
        xtrain, ytrain = balanced_sample(xtrain, ytrain, random_state=seed)
        xtrain = xtrain.drop("ID")
        
        # scale feature matrix
        xtrain = self.scaler.fit_transform(xtrain) # type: ignore
        self.model.fit(xtrain, ytrain)
        self.fitted = True
        self.scaled = True
        

    def predict(self, x, id_test=None):
        # Check that we received the expected arguments for this model
        assert self.fitted, "Model must be fitted before prediction"
        assert self.scaled, "Scaling model must be fitted before prediction"

        # Preprocess the new data
        if id_test is not None:
            idx = x["ID"].is_in(id_test)
            xtest = self.scaler.transform(x.filter(idx).drop("ID"))
        else:
            xtest = self.scaler.transform(x.drop("ID"))

        ypred = self.model.predict_proba(xtest)
        
        # Create a dictionary with ID column
        result_dict = {"ID": x.filter(idx)["ID"]}

        # Add a column for each class probability
        for i in range(ypred.shape[1]):
            result_dict[f"Ypred{i}"] = ypred[:, i] 

        return pl.DataFrame(result_dict)

    def get_importance(self, x: pl.DataFrame):
        # Extract feature importance from the model
        weights = self.model.coef_[0]
        features = x.drop("ID").columns
        importance = pd.Series(weights, index=features)
        importance = importance.sort_values(ascending=False)
        
        return importance
    
        
class BinaryLogistic(Logistic):
    """Logistic regression model for binary classification. Additional kwargs as specified in `sklearn.linear_model.LogisticRegression`."""
    def predict(self, x, id_test=None):
        # Check arguments and delegate to parent
        result_df = super().predict(x=x, id_test=id_test)
        result_df = result_df.select(["ID", "Ypred1"])
        return result_df.rename({"Ypred1": "Ypred"})


# --- Random forest classification models --- 
class RandomForest(SKLearnModel):
    """Random forest model for binary classification. Additional kwargs as specified in `sklearn.ensemble.RandomForestClassifier`."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize model
        model_params = self.params.get("model")
        if model_params is not None and "RandomForest" in model_params:
            kwargs = model_params["RandomForest"]
        elif model_params is not None and "BinaryRandomForest" in model_params:
            kwargs = model_params["BinaryRandomForest"]
        else:
            kwargs = {}

        self.model = RandomForestClassifier(**kwargs)
        self.fitted = False

    def fit(self, x, y, id_train=None):
        # Check that we received the expected arguments for this model

        # Filter to training ids if provided
        if id_train is not None:
            idx = x["ID"].is_in(id_train)
            xtrain = x.filter(idx)
            ytrain = y[0][idx]
        else:
            xtrain = x
            ytrain = y[0]

        # sample balance
        seed = self.params.get("seed", 47)
        xtrain, ytrain = balanced_sample(xtrain, ytrain, random_state=seed)
        xtrain = xtrain.drop("ID")

        # fit model
        self.model.fit(xtrain, ytrain)
        self.fitted = True

    def predict(self, x, id_test=None):
        # Check that we received the expected arguments for this model
        assert self.fitted, "Model must be fitted before prediction"

        # Preprocess the new data
        if id_test is not None:
            idx = x["ID"].is_in(id_test)
            xtest = x.filter(idx).drop("ID")
            id_col = x.filter(idx)["ID"]
        else:
            xtest = x.drop("ID")
            id_col = x["ID"]

        ypred = self.model.predict(xtest)
        return pl.DataFrame({"ID": id_col, "Ypred": ypred})
    
    def get_importance(self, x: pl.DataFrame):
        # Extract feature importance from the model
        assert self.fitted, "Model must be fitted before getting importance"
        importance = pd.Series(
            self.model.feature_importances_,
            index=x.drop("ID").columns
        ).sort_values(ascending=False)
        return importance

        
class BinaryRandomForest(RandomForest):
    """Random forest model for binary classification. Additional kwargs as specified in `sklearn.ensemble.RandomForestClassifier`."""
    def predict(self, x, id_test=None):
        # Check arguments and delegate to parent
        return super().predict(x=x, id_test=id_test)


# --- MultiAntibody classification models ---
class MultiAntibody(PyTorchModel):
    """Wrapper for multimodal PyTorch models to work with existing framework."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.fitted = False
    
    def fit(self, data_loader):
        if data_loader is None:
            raise ValueError("MultiAntibody requires a data_loader.")

        # Initialize model - requires reading out feature dims from data loader
        model_params = self.params.get("model")
        if model_params is not None and "MultiAntibody" in model_params:
            kwargs = model_params["MultiAntibody"]
        else:
            kwargs = {"train": TrainConfig(), "model": ModelConfig()}

        model_config = kwargs.get("model", ModelConfig())
        model_config.antibody_feature_dims = data_loader._get_feature_dims() 
        self.model = MultiAntibodyClassifier(**vars(model_config))

        # Train model 
        train_config = kwargs.get("train", TrainConfig())
        train(self.model, data_loader, train_config)
        self.fitted = True

    def predict(self, data_loader):
        assert self.fitted, "Model must be fitted before prediction"
        assert self.model is not None, "Model must be initialized"
        
        self.model.eval()
        all_probs = []
        all_labels = []
        all_lines = []
        device = next(self.model.parameters()).device
        data_loader.mode = "eval"
        
        with torch.no_grad():
            for batch in data_loader:
                if batch is None:
                    continue
                x_dict = {ab: batch[ab][0].to(device) for ab in batch}
                y_line = batch[list(batch.keys())[0]][1].to(device)
                cl = batch[list(batch.keys())[0]][-1]
                _, line_logits = self.model(x_dict)
                probs = torch.softmax(line_logits, dim=1)
                all_probs.append(probs.cpu())
                all_labels.append(y_line.cpu())

                if not isinstance(cl, list):
                    cl = [cl]
                
                all_lines.extend(cl)
            
        # Merge results into DF
        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        preds = pd.DataFrame(all_probs.numpy())
        preds.columns = [
            f"Class_{i}" for i in range(self.model.line_head.fc.out_features)
        ]
        
        preds["CellLines"] = all_lines
        preds["True"] = all_labels.numpy()
        preds = preds.sort_values(by="Class_0", ascending=False)
        return preds

    def _get_importance(self):
        # Not implemented for deep models
        return None