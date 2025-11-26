import polars as pl
import numpy as np
import pandas as pd
from maps.model_utils import cor, GradientReversal, l2_normalize


from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal


class Delearner():
    """Feed forward neural network model with losses for binary classification and reverse gradient loss for attributes to delearn."""    
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

