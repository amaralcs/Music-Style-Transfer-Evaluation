import os
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.train import Checkpoint, CheckpointManager
from tensorflow.keras.layers import Layer, Input, Conv2D, Lambda, ReLU


class InstanceNorm(Layer):
    """Custom implementation of Layer Normalization"""

    def __init__(self, epsilon=1e-5):
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def call(self, X):
        ## TODO: review InstanceNorm and refactor this
        scale = tf.Variable(
            initial_value=np.random.normal(1.0, 0.02, X.shape[-1:]),
            trainable=True,
            name="SCALE",
            dtype=tf.float32,
        )
        offset = tf.Variable(
            initial_value=np.zeros(X.shape[-1:]),
            trainable=True,
            name="OFFSET",
            dtype=tf.float32,
        )
        mean, variance = tf.nn.moments(X, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (X - mean) * inv
        return scale * normalized + offset


class ResNetBlock(Layer):
    """Implements a ResNet block with convolutional layers"""

    def __init__(
        self,
        n_units,
        kernel_size=3,
        strides=1,
        activation="relu",
        kernel_initializer=tf.random_normal_initializer(0, 0.2),
    ):
        """A custom implementation of a ResNet block, as used by the original authors.

        Parameters
        ----------
        n_units : int
            The number of units in each convolutional layer.
        kernel_size : int, Optional
        strides : int, Optional
        activation : str, Optional
            Name of activation to use for the block.
        kernel_initializer : tf.Initializer, Optional
            The initialization function to use.
        """
        super(ResNetBlock, self).__init__()
        self.n_units = n_units
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.strides = strides
        self.pad_size = (self.kernel_size - 1) // 2
        self.padding = "valid"
        self.activation = keras.activations.get(activation)

    def call(self, X):
        """
        Parameters
        ----------
        X : tf.tensor
            Input tensor.
        """
        y = X

        for idx in range(2):
            y = Lambda(
                input_padding,
                arguments={"pad_size": self.pad_size},
                name=f"padding_{idx}",
            )(y)
            y = Conv2D(
                self.n_units,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                kernel_initializer=self.kernel_initializer,
                use_bias=False,
            )(y)
            y = InstanceNorm()(y)

        return self.activation(y + X)


def input_padding(X, pad_size=3):
    """A custom padding implemented by the authors for some of the inputs

    Parameters
    ----------
    X : tf.tensor
        Input tensor to pad
    pad_size : int, Optional
        The size of the padding.
    """
    return tf.pad(
        X, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode="REFLECT"
    )


def create_checkpoint(
    genre_A,
    genre_B,
    epochs,
    checkpoint_dir="checkpoints",
    max_to_keep=5,
    **check_kwargs,
):
    dir_name = "{}2{}-{}e-{}".format(
        genre_A, genre_B, epochs, datetime.now().strftime("%Y-%m-%d")
    )
    os.makedirs(dir_name, exist_ok=True)

    checkpoint = Checkpoint(**check_kwargs)
    manager = CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=max_to_keep)
    return manager
