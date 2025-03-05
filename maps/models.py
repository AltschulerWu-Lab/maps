import polars as pl
import numpy as np
import pandas as pd
from typing import List
from maps.model_utils import *

from tensorflow.keras.models import Model
from tensorflow.keras import layers, initializers, constraints, losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class BaseModel():
    def __init__(self, params):
        self.params = params
        self.model_type = list(params.get("model"))[0]
        self.kwargs = params.get("model")[self.model_type] 
        self.model = eval(self.model_type)(**self.kwargs)
        
    def fit(self, x, y, id_train):
        return self.model._fit(x, y, id_train)
        
    def predict(self, fitted, x, id_test):
        return self.model._predict(fitted, x, id_test)
    
    def get_importance(self, fitted, x):
        return self.model._get_importance(fitted, x)
    
    
class Logistic():
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _fit(self, x: pl.DataFrame, y: List, id_train: pl.Series):
    
        # Convert Polars DataFrame to NumPy array
        idx = x["ID"].is_in(id_train)
        xtrain = x.filter(idx).drop("ID")
        scaler = StandardScaler()
        xtrain = scaler.fit_transform(xtrain)
        self.scaler = scaler
        ytrain = y[0][idx]
        
        model = LogisticRegression(**self.kwargs)
        model.fit(xtrain, ytrain)
        
        return {"model": model, "scaler": scaler}


    def _predict(self, model, x: pl.DataFrame, id_test: pl.Series):
       
        # Preprocess the new data
        idx = x["ID"].is_in(id_test)
        xtest = model["scaler"].transform(x.filter(idx).drop("ID"))
        ypred = model["model"].predict_proba(xtest)[:, 1]
        return pl.DataFrame({"ID": x.filter(idx)["ID"], "Ypred": ypred})

    
    def _get_importance(self, model, x: pl.DataFrame):
        # Extract feature importance from the model
        weights = model["model"].coef_[0]
        features = x.drop("ID").columns
        importance = pd.Series(weights, index=features)
        importance = importance.sort_values(ascending=False)
        
        return importance


class Delearner():
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _fit(self, x: pl.DataFrame, y: np.ndarray, id_train: pl.Series):

        # Define parameterized layers
        kernel = RandomNormal(mean=0.0, stddev=0.05)
        embedding_dim = self.kwargs.get("embedding_dim", 32)
        
        fc_layer = layers.Dense(
            embedding_dim, 
            activation='relu', 
            kernel_initializer=kernel
        ) 
       
        learning_head_layer = layers.Dense(
            1, 
            activation='sigmoid', 
            kernel_initializer=kernel,
            name='learning_head'
        )
        
        delearning_head_layer = layers.Dense(
            1,
            activation='linear',
            kernel_initializer=kernel,
            name="delearning_head"
        )
                   
        # Initialize model architecture
        model_input = layers.Input(shape=(x.shape[1] - 1,))
        batch_norm = layers.BatchNormalization()(model_input)
        dropout = layers.Dropout(0.5)(batch_norm) 
        embedding = fc_layer(dropout)
        reversal = GradientReversal()(embedding)
        learning_head = learning_head_layer(embedding)
        delearning_head = delearning_head_layer(reversal)
     
        model = Model(
            inputs=model_input, 
            outputs=[learning_head, delearning_head]
        )
        
        # Compile the model for full learning
        model.compile(
            optimizer=Adam(), 
            loss={
                "learning_head": "binary_crossentropy", 
                "delearning_head": "mean_absolute_percentage_error"
            },
            loss_weights={
                "learning_head": 1.0,
                "delearning_head": 0.01
            },
            metrics={
                "learning_head": "accuracy", 
                "delearning_head": cor
            },
        )
        
        # Fit model
        idx = x["ID"].is_in(id_train)
        xtrain = l2_normalize(x.filter(idx))
        ytrain = [yy[idx].astype("float") for yy in y]
        
        model.fit(xtrain, ytrain, **self.kwargs)
        return model


    def _predict(self, model: Model, x: pl.DataFrame, id_test: pl.Series):
        
        # Preprocess the new data
        idx = x["ID"].is_in(id_test)
        xtest = l2_normalize(x.filter(idx))    
        ypred = model.predict(xtest)[0][:,0]
        return pl.DataFrame({"ID": x.filter(idx)["ID"], "Ypred": ypred})

    
    def _get_importance(self, model: Model, x: pl.DataFrame):
        # Extract feature importance from the model    
        d = (len(x.columns) - 1)
        x_zeros = np.zeros((d, d))
        x_identity = np.eye(d)
        
        ypred_zeros = model.predict(x_zeros)[0]
        ypred_identity = model.predict(x_identity)[0]
        importance = ypred_identity - ypred_zeros
        
        features = x.drop("ID").columns
        importance = pd.Series(importance[:, 0], index=features)
        importance = importance.sort_values(ascending=False)
        
        return importance





if False:
    #### Prototyping models ####
    def nmflr(n, m, k):
        # Non-negative matrix factorization + logistic regression model
        # TODO: need to build constructors that take sample and fit dict weights, then get predicitons from fitted weights
        
        # Embedding layers for cell and feature IDs
        cell_id_input = layers.Input(shape=(1,), name="cell_id")
        feature_id_input = layers.Input(shape=(1,), name="feature_id")
        
        c_embedding = layers.Embedding(
            input_dim=n, 
            output_dim=k, 
            embeddings_initializer=initializers.RandomUniform(minval=0, maxval=1),
            embeddings_constraint=constraints.NonNeg()                           
        )(cell_id_input)
        
        f_embedding = layers.Embedding(
            input_dim=m, 
            output_dim=k,
            embeddings_initializer=initializers.RandomUniform(minval=0, maxval=1),
            embeddings_constraint=constraints.NonNeg()
        )(feature_id_input)
        
        c_embedding = layers.Flatten()(c_embedding) # (n, K)
        f_embedding = layers.Flatten()(f_embedding) # (m, K) 
        x = layers.Dot(axes=(1, 1), name="embedded")([c_embedding, f_embedding])
        
        # Logistic regression layer defined on embeddings
        dense = layers.Dense(16, activation='sigmoid')(c_embedding)
        norm = layers.BatchNormalization()(dense)
        response = layers.Dense(1, activation='sigmoid', name="response")(norm)
                
        model = Model(
            inputs=[cell_id_input, feature_id_input], 
            outputs=[x, response]
        )
       
        return model 
        model.compile(
            optimizer=Adam(), 
            loss={
                'embedded':losses.MeanSquaredError(),
                'response':losses.BinaryCrossentropy()
            }     
        )
        
        return model


    def flatten_with_indices(df: pl.DataFrame):
            # Normalize the NumPy array
            df_norm = l2_normalize(df)
            
            # Flatten the NumPy array
            values = df_norm.flatten()
            
            # Generate row and column indices
            row_indices, column_indices = np.indices(df_norm.shape)
            row_indices = row_indices.flatten()
            column_indices = column_indices.flatten()
            
            return row_indices, column_indices, values

    def logistic_nmf(x: pl.DataFrame, y: np.ndarray, **kwargs) -> Sequential:
        # Fits keras model with dual loss: non-negative matrix factorization and logistic regression
        
        # Convert Polars DataFrame to NumPy array
        x_norm = l2_normalize(x)
        
        # Define keras non-negative matrix factorization model
        input_layer = Input(shape=(x_norm.shape[1],))
        