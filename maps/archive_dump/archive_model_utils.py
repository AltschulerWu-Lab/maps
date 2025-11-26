import tensorflow as tf
from tensorflow import keras
import polars as pl
import numpy as np
import tensorflow as tf

    
def l2_normalize(df: pl.DataFrame) -> np.ndarray:
    # Convert Polars DataFrame to NumPy array
    np_array = df.drop("ID").to_numpy()
    
    # L2 normalize each column
    norms = np.linalg.norm(np_array, axis=0)
    normalized_array = np_array / norms
    
    return normalized_array


def cor(y_true, y_pred):
    """ Pearson correlation loss for adversarial training """
    yt = y_true - tf.reduce_mean(y_true)
    yp = y_pred - tf.reduce_mean(y_pred)
    num = tf.reduce_sum(yt * yp)
    denom = tf.sqrt(tf.reduce_sum(yt ** 2)) * tf.sqrt(tf.reduce_sum(yp ** 2))
    return num / (denom + 1e-7)


def rmse(y_true, y_pred):
    """ Root mean squared error """
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))
    var = tf.reduce_mean(tf.square(y_true - tf.reduce_mean(y_true)))
    return rmse / (var + 1e-7)


class GradientReversal(keras.layers.Layer):
    """Flip the sign of gradient during training.
    based on https://github.com/michetonu/gradient_reversal_keras_tf
    ported to tf 2.x
    """

    def __init__(self, λ=1, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.λ = λ

    @staticmethod
    @tf.custom_gradient
    def reverse_gradient(x, λ):
        # @tf.custom_gradient suggested by Hoa's comment at
        # https://stackoverflow.com/questions/60234725/how-to-use-gradient-override-map-with-tf-gradienttape-in-tf2-0
        return tf.identity(x), lambda dy: (-dy, None)

    def call(self, x):
        return self.reverse_gradient(x, self.λ)

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super(GradientReversal, self).get_config() | {'λ': self.λ}