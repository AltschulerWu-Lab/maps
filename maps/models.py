"""Model wrappers and implementations used by the MAP pipeline.

This module provides a lightweight abstraction for classification models used
to compute MAP scores. Subclasses of :class:`BaseModel` implement fitting,
prediction, and feature-importance extraction for a variety of model types
including scikit-learn and PyTorch based implementations.

Module variables:

- ``DEFAULT_SEED`` (int): Default random seed used when not specified in params.
"""

#: Default random seed used when not specified in params.
DEFAULT_SEED = 47
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
    """Base class for all classification models.

    This class serves as a wrapper for fitting models, generating predictions, and
    computing feature importances. Subclasses should implement specific model logic
    in ``fit()``, ``predict()``, and ``get_importance()`` methods.

    Attributes:

    - params (dict): Model parameters and configuration.
    - model: The underlying model object (implementation-specific).
    - fitted (bool): Whether the model has been fitted to data.
    """
    def __init__(self, **kwargs):
        """Initialize a BaseModel instance.

        Args:
            **kwargs: Arbitrary keyword arguments for model configuration.
        """
        self.params = kwargs
        self.model = None
        self.fitted = False
    
    def fit(self, x=None, y=None, id_train=None, data_loader=None):
        """Fit the model to training data.

        Args:
            x (pl.DataFrame, optional): Feature matrix. Defaults to None.
            y (np.ndarray, optional): Response vector. Defaults to None.
            id_train (list, optional): IDs of training samples. Defaults to None.
            data_loader (DataLoader, optional): PyTorch DataLoader for batch training.
                Defaults to None.

        Returns:
            None

        Note:
            This method should be implemented by subclasses.
        """
        raise NotImplementedError()
    
    def predict(self, x=None, y=None, id_test=None, data_loader=None):
        """Generate predictions for test data.

        Args:
            x (pl.DataFrame, optional): Feature matrix. Defaults to None.
            y (np.ndarray, optional): Response vector. Defaults to None.
            id_test (list, optional): IDs of test samples. Defaults to None.
            data_loader (DataLoader, optional): PyTorch DataLoader for batch prediction.
                Defaults to None.

        Returns:
            pl.DataFrame: Predictions with ID column and probability columns.

        Note:
            This method should be implemented by subclasses.
        """
        raise NotImplementedError()
    
    def get_importance(self, x=None):
        """Extract feature importances from the fitted model.

        Args:
            x (pl.DataFrame, optional): Feature matrix used for training. Defaults to None.

        Returns:
            pd.Series or None: Feature importances sorted in descending order, or None
                if not implemented.

        Note:
            This method should be implemented by subclasses.
        """
        raise NotImplementedError()
        
class SKLearnModel(BaseModel):
    """Base class for all scikit-learn based models.
    
    Attributes:
        model_type (str): Type identifier set to "sklearn".
    """
    def __init__(self, **kwargs):
        """Initialize an SKLearnModel instance.
        
        Args:
            **kwargs: Arbitrary keyword arguments passed to BaseModel.
        """
        super().__init__(**kwargs)        
        self.model_type = "sklearn"

class PyTorchModel(BaseModel):
    """Base class for all PyTorch based models.
    
    Attributes:
        model_type (str): Type identifier set to "pytorch".
    """
    def __init__(self, **kwargs):
        """Initialize a PyTorchModel instance.
        
        Args:
            **kwargs: Arbitrary keyword arguments for model configuration.
        """
        super().__init__(**kwargs)
        self.params = kwargs
        self.model_type = "pytorch"

# --- Logistic regression classification models ---
class Logistic(SKLearnModel):
    """Logistic regression model for multi-class classification.
    
    This model uses scikit-learn's LogisticRegression with feature scaling and
    balanced sampling during training.
    
    Attributes:
        scaler (StandardScaler): Scaler for feature normalization.
        model (LogisticRegression): The underlying logistic regression model.
        fitted (bool): Whether the model has been fitted.
        scaled (bool): Whether the scaler has been fitted.
    
    Example:
        >>> model = Logistic(model={"Logistic": {"max_iter": 1000}})
        >>> model.fit(x_train, y_train)
        >>> predictions = model.predict(x_test)
    """
    def __init__(self, **kwargs):
        """Initialize a Logistic model.
        
        Args:
            **kwargs: Model parameters. Can include a 'model' dict with 'Logistic'
                or 'BinaryLogistic' keys containing sklearn LogisticRegression parameters.
        """
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
        """Fit the logistic regression model.
        
        Args:
            x (pl.DataFrame): Feature matrix with ID column.
            y (list): List containing response array(s).
            id_train (list, optional): IDs of training samples. If None, uses all samples.
                Defaults to None.
        """
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
        """Generate probability predictions.
        
        Args:
            x (pl.DataFrame): Feature matrix with ID column.
            id_test (list, optional): IDs of test samples. If None, uses all samples.
                Defaults to None.
        
        Returns:
            pl.DataFrame: Predictions with ID and probability columns (Ypred0, Ypred1, etc.).
        
        Raises:
            AssertionError: If model has not been fitted or scaler has not been fitted.
        """
        # Check that we received the expected arguments for this model
        assert self.fitted, "Model must be fitted before prediction"
        assert self.scaled, "Scaling model must be fitted before prediction"

        # Preprocess the new data
        if id_test is not None:
            idx = x["ID"].is_in(id_test)
            xtest = self.scaler.transform(x.filter(idx).drop("ID"))
        else:
            idx = pd.Series([True] * x.shape[0])
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
    """Logistic regression for binary classification.
    
    Extends the Logistic class but returns only the probability for the positive class.
    
    Example:
        >>> model = BinaryLogistic(model={"BinaryLogistic": {"C": 1.0}})
        >>> model.fit(x_train, y_train)
        >>> predictions = model.predict(x_test)  # Returns Ypred column only
    """
    def predict(self, x, id_test=None):
        """Generate binary probability predictions.
        
        Args:
            x (pl.DataFrame): Feature matrix with ID column.
            id_test (list, optional): IDs of test samples. Defaults to None.
        
        Returns:
            pl.DataFrame: Predictions with ID and Ypred columns (positive class probability).
        """
        # Check arguments and delegate to parent
        result_df = super().predict(x=x, id_test=id_test)
        result_df = result_df.select(["ID", "Ypred1"])
        return result_df.rename({"Ypred1": "Ypred"})


# --- Random forest classification models --- 
class RandomForest(SKLearnModel):
    """Random forest classifier for multi-class classification.
    
    This model uses scikit-learn's RandomForestClassifier with balanced sampling
    during training.
    
    Attributes:
        model (RandomForestClassifier): The underlying random forest model.
        fitted (bool): Whether the model has been fitted.
    
    Example:
        >>> model = RandomForest(model={"RandomForest": {"n_estimators": 100}})
        >>> model.fit(x_train, y_train)
        >>> predictions = model.predict(x_test)
    """
    def __init__(self, **kwargs):
        """Initialize a RandomForest model.
        
        Args:
            **kwargs: Model parameters. Can include a 'model' dict with 'RandomForest'
                or 'BinaryRandomForest' keys containing sklearn RandomForestClassifier
                parameters.
        """
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
        """Fit the random forest model.
        
        Args:
            x (pl.DataFrame): Feature matrix with ID column.
            y (list): List containing response array(s).
            id_train (list, optional): IDs of training samples. If None, uses all samples.
                Defaults to None.
        """
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
        """Generate class predictions.
        
        Args:
            x (pl.DataFrame): Feature matrix with ID column.
            id_test (list, optional): IDs of test samples. If None, uses all samples.
                Defaults to None.
        
        Returns:
            pl.DataFrame: Predictions with ID and Ypred columns (predicted class).
        
        Raises:
            AssertionError: If model has not been fitted.
        """
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
        """Extract feature importances from the random forest.
        
        Args:
            x (pl.DataFrame): Feature matrix with ID column and feature columns.
        
        Returns:
            pd.Series: Feature importances sorted in descending order.
        
        Raises:
            AssertionError: If model has not been fitted.
        """
        # Extract feature importance from the model
        assert self.fitted, "Model must be fitted before getting importance"
        importance = pd.Series(
            self.model.feature_importances_,
            index=x.drop("ID").columns
        ).sort_values(ascending=False)
        return importance

        
class BinaryRandomForest(RandomForest):
    """Random forest classifier for binary classification.
    
    Extends the RandomForest class for binary classification tasks.
    
    Example:
        >>> model = BinaryRandomForest(model={"BinaryRandomForest": {"n_estimators": 50}})
        >>> model.fit(x_train, y_train)
        >>> predictions = model.predict(x_test)
    """
    def predict(self, x, id_test=None):
        """Generate binary class predictions.
        
        Args:
            x (pl.DataFrame): Feature matrix with ID column.
            id_test (list, optional): IDs of test samples. Defaults to None.
        
        Returns:
            pl.DataFrame: Predictions with ID and Ypred columns.
        """
        # Check arguments and delegate to parent
        return super().predict(x=x, id_test=id_test)


# --- MultiAntibody classification models ---
class MultiAntibody(PyTorchModel):
    """Multi-modal transformer model for multi-antibody imaging data.
    
    This model processes data from multiple antibody stains using a transformer-based
    architecture. It requires a specialized DataLoader that provides batched multi-modal
    data.
    
    Attributes:
        model (MultiAntibodyClassifier): The underlying PyTorch model.
        fitted (bool): Whether the model has been fitted.
    
    Example:
        >>> model = MultiAntibody(model={"MultiAntibody": {
        ...     "model": {"n_classes": 2, "d_model": 16, "n_layers": 1},
        ...     "train": {"n_epochs": 100}
        ... }})
        >>> model.fit(data_loader=train_loader)
        >>> predictions = model.predict(data_loader=test_loader)
    """
    def __init__(self, **kwargs):
        """Initialize a MultiAntibody model.
        
        Args:
            **kwargs: Model parameters. Should include a 'model' dict with 'MultiAntibody'
                key containing 'model' (ModelConfig params) and 'train' (TrainConfig params).
        """
        super().__init__(**kwargs)
        self.model = None
        self.fitted = False
    
    def fit(self, data_loader):
        """Fit the multi-antibody model.
        
        Args:
            data_loader: DataLoader providing multi-modal batched data. Must implement
                _get_feature_dims() method to determine input dimensions.
        
        Raises:
            ValueError: If data_loader is None.
        """
        if data_loader is None:
            raise ValueError("MultiAntibody requires a data_loader.")

        # Initialize model - requires reading out feature dims from data loader
        model_params = self.params.get("model")
        if model_params is not None and "MultiAntibody" in model_params:
            kwargs = model_params["MultiAntibody"]
        else:
            print("No model params found for MultiAntibody, using defaults.")
            kwargs = {"train": TrainConfig(), "model": ModelConfig()}

        model_config = ModelConfig(**kwargs.get("model", {}))
        model_config.antibody_feature_dims = data_loader._get_feature_dims() 
        self.model = MultiAntibodyClassifier(**vars(model_config))

        # Train model 
        train_config = TrainConfig(**kwargs.get("train", {}))
        train(self.model, data_loader, train_config)
        self.fitted = True

    def predict(self, data_loader):
        """Generate probability predictions for multi-antibody data.
        
        Args:
            data_loader: DataLoader providing multi-modal batched test data.
        
        Returns:
            pl.DataFrame: Predictions with columns for each class probability (Class_0,
                Class_1, etc.), CellLines, and True labels.
        
        Raises:
            AssertionError: If model has not been fitted or model is None.
        """
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
        return pl.DataFrame(preds)

    def _get_importance(self):
        """Get feature importances.
        
        Note:
            Feature importance extraction is not implemented for deep learning models.
        
        Returns:
            None: Always returns None for this model type.
        """
        # Not implemented for deep models
        return None