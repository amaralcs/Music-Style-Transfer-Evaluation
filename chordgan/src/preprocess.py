from cgitb import handler
from multiprocessing.context import get_spawning_popen
import numpy as np
import pretty_midi
import reverse_pianoroll
import convert
import argparse
from glob import glob
import os
import logging

logger = logging.Logger(
    "preprocessing_logger",
    level=logging.INFO,
    handler=logging.StreamHandler()
)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type="str",
        default="data/chordGan",
        help="Path to where the data is stored.",
    )
    parser.add_argument(
        "genre",
        type="str",
        default="pop",
        choices=["pop", "classical", "jazz"],
        help="Genre of dataset to be loaded.",
    )
    parser.add_argument(
        "outpath",
        type="str",
        default="test/data",
        help="Location where the outputs will be saved.",
    )
    return parser.parse_args(args)


def get_songs(path):
    files = glob.glob("{}/*.mid*".format(path))
    songs = []
    fnames = []
    logger.info(f"Reading {len(files)} files from {path}:")
    for f in files:
        try:
            data = pretty_midi.PrettyMIDI(f)
            song = data.get_piano_roll(fs=16)
            song = convert.forward(song)
            # song = np.transpose(song) - if your code matrices aren't working, try uncommenting this. the convert.py file might not be updated
            songs.append(song)
            fnames.append(f)
        except Exception as e:
            logger.warn(f"  there was an error loading {f}:")
            raise e
    return songs, fnames


# custom function to extract chroma features from song
def get_chromas(songs, chroma_dims=12, n_notes=78):
    """
    chroma_dims : int
        The dimension of the chromas (12 notes in an equal tempered scale)
    n_notes : int
        The number of notes in the MIDI format
    """
    chromas = []
    logger.info("Generating chroma features from loaded songs")
    for song in songs:
        chroma = np.zeros(shape=(song.shape[0], chroma_dims))  # 12 chroma values
        for i in np.arange(song.shape[0]):
            for j in np.arange(n_notes):
                if song[i][j] > 0:
                    chroma[i][np.mod(j, chroma_dims)] += 1
        chromas.append(chroma)
    return chromas


def main(args):
    args = parse_args(args)

    fpath = args.fpath
    genre = args.genre
    full_path = os.path.join(fpath, genre)

    songs, names = get_songs(full_path)
    chromas = get_chromas(songs)
    
    logger.info("Complete.")


if __name__ == "__main__":
    main(args[1:])
