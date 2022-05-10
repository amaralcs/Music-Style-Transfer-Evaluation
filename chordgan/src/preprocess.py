import numpy as np
import pretty_midi
# import convert
from glob import glob
import os
import logging
import tensorflow as tf
from functools import reduce

logger = logging.getLogger("preprocessing_logger")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def get_chroma(song, chroma_dims=12, n_notes=84):
    """Creates the chromagram of a given song.

    Implementation taken from:
    https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S1_SpecLogFreq-Chromagram.html

    Parameters
    ----------
    song : pypianoroll
        Piano roll of a given song
    chroma_dims : int, Optional
        How many chroma dimensions to split.
    n_notes : int, Optional
        The number of notes in the range of the clipped midi.

    Returns
    -------
    chroma
    """
    chroma = np.zeros(shape=(song.shape[0], chroma_dims))
    p = np.arange(n_notes)

    # for each pitch class
    for c in range(12):
        # create a mask for note indices that belong to this class
        mask = (p % 12) == c
        # calculate the chroma as the sum of all pitches in this class
        chroma[:, c] = song[:, mask].sum(axis=1)
    return chroma


def reshape_step(midi, n_sequences, total_timesteps):
    """Trims the song 

    Parameters
    ----------
    midi : np.array
        Song or chroma pianoroll.
    n_sequences : int
        Number of sequences of size `n_bars` * `fs` that can be fit exactly in the given
        input midi.
    total_timesteps: int
        Total number of timesteps in the number of given bars times # of notes per bar.

    Returns
    -------
    input_
    """
    return midi[: n_sequences * total_timesteps]  # discard any extra timesteps


def clip_midi(midi, low=24, high=108):
    """Clips the given midi to a specified range of notes.

    Clipping is done along the first axis and the default range is C1 (24) - C8 (108).

    Parameters
    ----------
    midi : np.array
        The midi array to clip.
    low : int
    high : int

    Returns
    -------
    np.array
    """
    return midi[low:high, :]


def normalize_velocities(midi):
    """Sets all non-zero velocities of the midi file to 1.

    Parameters
    ----------
    midi : np.array
        The midi array to normalize.

    Returns
    -------
    np.array
    """
    idx_x, idx_y = midi.nonzero()
    midi[idx_x, idx_y] = 1
    return midi


def get_songs(path, n_bars=4, fs=16):
    """Loads midi files from the given path and converts them to piano roll format.

    Parameters
    ----------
    path : str
        Path to directory containing MIDI files
    n_bars : int
        The number of bars that each song sequence will have.
    fs : int
        The sampling rate when generating the piano roll. It is also the number of 
        notes that each bar will contain.

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

    total_timesteps = n_bars * fs
    for f in files:
        try:
            data = pretty_midi.PrettyMIDI(f)
            song = data.get_piano_roll(fs=fs)
            song = clip_midi(song)
            song = normalize_velocities(song)
            song = song.T  # Transpose the song to (n_notes, note_range)

            chroma = get_chroma(song)

            # calculate the number of sequences of size n_bars * fs 
            # we can use to split this song into
            n_sequences = song.shape[0] // total_timesteps

            # Then use it to reshape the song and chromas
            # Note: At the moment this simply reshapes the song and treats it as a whole
            # Another approach to be tried is to split the song into `n_sequences` and add 
            # each sequence individually to the training set. TODO
            song = reshape_step(song, n_sequences, total_timesteps)
            chroma = reshape_step(chroma, n_sequences, total_timesteps)

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


def join_datasets(song_ds, chroma_ds, shuffle, shuffle_buffer=10000):
    """Joins two given datasets to create inputs of the form ((s1, c1), (s2, c2), ...)

    Parameters
    ----------
    song_ds : tf.data.Dataset
        Dataset with songs.
    chroma_ds : tf.data.Dataset
        Dataset with song chromas.
    shuffle : bool
        Whether to shuffle the resulting dataset. Setting it to False is useful when
        processing a dataset for genre transfer as the songs will retain their order
        and can be identified with the `names` output.
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
    ds = tf.data.Dataset.zip((song_ds, chroma_ds)).map(
        lambda song, chroma: (
            tf.expand_dims(song, axis=0),
            tf.expand_dims(chroma, axis=0),
        )
    )
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)

    return ds.prefetch(1)


def load_data(fpath, genre, shuffle=True):
    """Loads and preprocess the data as required for ChordGAN

    Parameters
    ----------
    fpath : str
        Path to where the data is stored.
    genre : str
        Genre of dataset to be loaded.
    shuffle : bool
        Whether to shuffle the resulting dataset. Setting it to False is useful when
        processing a dataset for genre transfer as the songs will retain their order
        and can be identified with the `names` output.

    Returns
    -------
    tf.data.Dataset
    """

    full_path = os.path.join(fpath, genre)

    songs, names, chromas = get_songs(full_path)
    song_ds = create_dataset(songs)
    chromas_ds = create_dataset(chromas)

    logger.info("Complete.")
    return join_datasets(song_ds, chromas_ds, shuffle), (songs, names)
