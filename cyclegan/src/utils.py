import os
import logging
import numpy as np
from glob import glob
from datetime import datetime
from functools import reduce

import tensorflow as tf
from tensorflow import keras
from tensorflow.train import Checkpoint, CheckpointManager
from tensorflow.keras.layers import Layer, Input, Conv2D, Lambda, ReLU
from tensorflow.data import Dataset


logger = logging.getLogger("utils_logger")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class InstanceNorm(Layer):
    """Custom implementation of Layer Normalization"""

    def __init__(self, epsilon=1e-5):
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = tf.Variable(
            initial_value=np.random.normal(1.0, 0.02, input_shape[-1:]),
            trainable=True,
            name="SCALE",
            dtype=tf.float32,
        )
        self.offset = tf.Variable(
            initial_value=np.zeros(input_shape[-1:]),
            trainable=True,
            name="OFFSET",
            dtype=tf.float32,
        )
        super(InstanceNorm, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (inputs - mean) * inv
        return self.scale * normalized + self.offset


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

    def build(self, input_shape):
        self.padding_1 = Lambda(
            input_padding, arguments={"pad_size": self.pad_size}, name=f"padding_1"
        )
        self.padding_2 = Lambda(
            input_padding, arguments={"pad_size": self.pad_size}, name=f"padding_1"
        )
        self.conv2d_1 = Conv2D(
            self.n_units,  # input_shape[-1], # TODO: Check whether to use this or `self.n_units`
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            use_bias=False,
            name="conv2D_1",
        )
        self.conv2d_2 = Conv2D(
            self.n_units,  # input_shape[-1], # TODO: Check whether to use this or `self.n_units`
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            use_bias=False,
            name="conv2D_2",
        )
        self.instance_norm = InstanceNorm()

        super(ResNetBlock, self).build(input_shape)

    def call(self, inputs):
        """
        Parameters
        ----------
        inputs : tf.tensor
            Input tensor.
        """
        X = self.padding_1(inputs)
        X = self.conv2d_1(X)
        X = self.instance_norm(X)
        X = self.padding_2(X)
        X = self.conv2d_2(X)
        X = self.instance_norm(X)
        return self.activation(X + inputs)


def input_padding(X, pad_size=3):
    """A custom padding implemented by the authors for some of the inputs

    Parameters
    ----------
    X : tf.tensor
        Input tensor to pad
    pad_size : int, Optional
        The size of the padding.

    Returns
    -------
        Input tenser with paddings of shape (pad_size, pad_size) applied to the
        2nd and 3rd dimensions.
    """
    return tf.pad(
        X, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode="REFLECT"
    )


def create_dataset(tensor_list):
    """Converts the a list of numpy arrays into a tensorflow dataset.

    Parameters
    ----------
    songs : List
        List of numpy arrays. That is, the piano roll representations of songs/chromas

    Returns
    -------
    tf.data.Dataset
    """
    logger.info("Creating TF dataset from the loaded phrases")
    datasets = [tf.data.Dataset.from_tensors(tensor) for tensor in tensor_list]
    return reduce(lambda ds1, ds2: ds1.concatenate(ds2), datasets)


def load_np_phrases(path, sample_size, set_type="train"):
    """Loads the preprocessed numpy phrases from a given path.

    Parameters
    ----------
    path : str
        Path to the prepared phrases.
    set_type: str, Optional
        Whether to load from train/test folder.

    Returns
    -------
    Lis[np.array]
    """
    logger.info(f"Loading {set_type} phrases from {path}")
    fnames = glob(os.path.join(path, set_type, "*.*"))
    if len(fnames) == 0:
        logger.error(
            f"There was an error loading data from {path}: Are you sure the path is correct?",
            f" The current working directory is {os.getcwd()}",
        )
    if sample_size:
        fnames = fnames[:sample_size]
    return [np.load(fname).astype(np.float32) for fname in fnames]


def join_datasets(
    dataset_a, dataset_b, batch_size=16, shuffle=False, shuffle_buffer=50_000
):
    """Joins two given datasets to create inputs of the form ((a1, b1), (a2, b2), ...)

    Parameters
    ----------
    dataset_a : tf.data.Dataset
        Dataset with songs from genre A.
    dataset_b : tf.data.Dataset
        Dataset with songs from genre B.
    shuffle : bool, Optional
        Whether to shuffle the resulting dataset.
    shuffle_buffer : int, Optional
        The size of the shuffle buffer

    Returns
    -------
    tf.data.Dataset

    Note
    ----
    I don't think I need batching here, but I can include it
    if I use `tf.data.Dataset.bucket_by_sequence_length`.
    """
    logger.info("Joining datasets")
    ds = tf.data.Dataset.zip((dataset_a, dataset_b))
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)

    return ds.batch(batch_size).prefetch(1)


def parse_tfr_array(array):
    array_description = {"array": tf.io.FixedLenFeature([], tf.string)}
    logger.info(array)
    parsed_array = tf.io.parse_single_example(array, array_description)
    tensor = tf.io.parse_tensor(parsed_array["array"], out_type=tf.float32)
    logger.info(f"Parsed tensor: {tensor}")
    return Dataset.from_tensor_slices(tensor)


def load_tfrecords(path, set_type):
    logger.info(f"Loading {set_type} tfrecords from {path}")
    return (
        tf.data.TFRecordDataset.list_files(
            os.path.join(path, set_type, "*.tfrecord")
        )  # returns a dataset
        .interleave(
            parse_tfr_array,
            cycle_length=50,
        )
        .shuffle(15_000)
    )


def load_data(path_a, path_b, set_type, batch_size, shuffle=False, sample_size=500):
    """Helper function for loading the numpy phrases and converting them into a tensorflow dataset

    Parameters
    ----------
    path_a : str
        Path to where phrases of genre A are stored.
    path_b : str
        Path to where phrases of genre B are stored.
    set_type: str, Optional
        Whether to load from train/test folder.
    batch_size : int
        Batch size to use for the dataset.
    shuffle : bool, Optional
        Whether to shuffle the resulting dataset.
    sample_size : int
        If not-null, use only the defined number of samples from each of the datasets.
        This is useful for testing runs.

    Returns
    -------
    tf.data.Dataset
    """
    dataset_a = load_tfrecords(path_a, set_type)
    dataset_b = load_tfrecords(path_b, set_type)
    print(type(dataset_a))
    # X_a_train = load_np_phrases(path_a, sample_size, set_type)
    # dataset_a = create_dataset(X_a_train)
    # X_b_train = load_np_phrases(path_b, sample_size, set_type)
    # dataset_b = create_dataset(X_b_train)
    # dataset_a = load_file_dataset(path_a)
    # dataset_b = load_file_dataset(path_b)

    return join_datasets(dataset_a, dataset_b, batch_size, shuffle=shuffle)
