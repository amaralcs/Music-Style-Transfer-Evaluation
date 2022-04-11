import sys
import numpy as np
import pretty_midi
import reverse_pianoroll
import convert
import argparse
from glob import glob
import os
import logging
import tensorflow as tf
from functools import reduce

logger = logging.getLogger("preprocessing_logger")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def get_chroma(song, chroma_dims=12, n_notes=78):
    """Creates the chromagram of a given song.

    Parameters
    ----------
    song : pypianoroll
        Piano roll of a given song
    chroma_dims : int, Optional
        How many chroma dimensions to split.
    n_notes : 78
        The range of nots to use.

    Returns
    -------
    chroma
    """
    chroma = np.zeros(shape=(song.shape[0], chroma_dims))  # 12 chroma values
    for i in np.arange(song.shape[0]):
        for j in np.arange(n_notes):
            if song[i][j] > 0:
                chroma[i][np.mod(j, chroma_dims)] += 1
    return chroma


def reshape_step(input_, input_timesteps, n_timesteps=4):
    """Applies ChordGAN reshaping to a given input

    Parameters
    ----------
    input_ : pypianoroll
        Song or chroma pianoroll.
    input_timesteps : int
        Number of timesteps to reshape the song to. Varies depending with song length.
    n_timestes: int
        The number of bars to limit each song to.

    Returns
    -------
    input_
    """
    input_ = input_[: input_timesteps * n_timesteps]  # discard any extra timesteps
    input_ = input_.reshape([input_timesteps, input_.shape[1] * n_timesteps])
    return input_


def get_songs(path, n_timesteps=4):
    """Loads midi files from the given path and converts them to piano roll format.

    Parameters
    ----------
    path : str
        Path to directory containing MIDI files

    Returns
    -------
    songs : List
        List of songs, loaded in the piano roll format.
    fnames : List
        List of song names loaded.
    chromas :
    """
    files = glob("{}/*.mid*".format(path))
    songs, fnames, chromas = [], [], []
    for f in files:
        try:
            data = pretty_midi.PrettyMIDI(f)
            song = data.get_piano_roll(fs=16)
            song = convert.forward(song)
            chroma = get_chroma(song)

            # Reshaping steps
            song_timesteps = song.shape[0] // n_timesteps
            song = reshape_step(song, song_timesteps)
            chroma = reshape_step(chroma, song_timesteps)

            songs.append(song)
            chromas.append(chroma)
            fnames.append(f)
        except Exception as e:
            raise e
    return songs, fnames, chromas


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
    logger.info("Creating TF dataset from the loaded songs")
    datasets = [tf.data.Dataset.from_tensors(tensor) for tensor in tensor_list]
    return reduce(lambda ds1, ds2: ds1.concatenate(ds2), datasets)


def join_datasets(song_ds, chroma_ds, shuffle_buffer=10000):
    """Joins two given datasets to create inputs of the form ((s1, c1), (s2, c2), ...)

    Parameters
    ----------
    song_ds : tf.data.Dataset
        Dataset with songs.
    chroma_ds : tf.data.Dataset
        Dataset with song chromas.
    shuffle_buffer : int, Optional
        The size of the shuffle buffer

    Returns
    -------
    tf.data.Dataset

    Note
    ----
    I don't think I need batching here, but I can include it
    if I use `tf.data.Dataset.bucket_by_sequence_length`.

    The map function expanding dims essentially creates a batch of size 1 so that
    it fits the model inputs

    """
    return (
        tf.data.Dataset.zip((song_ds, chroma_ds))
        .shuffle(shuffle_buffer)
        .map(
            lambda song, chroma: (
                tf.expand_dims(song, axis=0),
                tf.expand_dims(chroma, axis=0),
            )
        )
        .prefetch(1)
    )


def load_data(fpath, genre):
    """Loads and preprocess the data as required for ChordGAN
    
    Parameters
    ----------
    fpath : str
        Path to where the data is stored.
    genre : str
        Genre of dataset to be loaded.

    Returns
    -------
    tf.data.Dataset
    """

    full_path = os.path.join(fpath, genre)

    songs, names, chromas = get_songs(full_path)
    song_ds = create_dataset(songs)
    chromas_ds = create_dataset(chromas)

    logger.info("Complete.")
    return join_datasets(song_ds, chromas_ds)
