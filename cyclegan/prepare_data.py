import logging
import os
import shutil
import json
import errno

import numpy as np
from pypianoroll import Multitrack, Track, from_pretty_midi
import pretty_midi

logger = logging.getLogger("preprocessing_logger")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

ROOT_PATH = "new_data"
GENRE_PATH = "Pop_Music_Midi"
TEST_RATIO = 0.1

converter_path = os.path.join(ROOT_PATH, "test/converter")
cleaner_path = os.path.join(ROOT_PATH, "test/cleaner")

def get_midi_path(root):
    """Return a list of paths to MIDI files in `root` (recursively)
    
    Parameters
    ----------
    root : str
        Path to directory with midi files.
    
    Returns
    -------
    List[str]
    """
    filepaths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if ".mid" in filename:
                filepaths.append(os.path.join(dirpath, filename))
    return filepaths

def make_sure_path_exists(path):
    """Create all intermediate-level directories if the given path does not exist.
    
    Parameters
    ----------
    path : str
        Name of folder to create.
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise exception


def get_midi_info(midi_obj):
    """Return stats of the loaded midi object.

    The particular stats we are interested in are:
        - first_beat_time
        - num_time_signature_changes (was there changes in the time signature?)
        - time_signature (None, if there were changes in the signature)
        - tempo
    
    Parameters
    ----------
    midi_obj : pretty_midi.PrettyMIDI
        The pretty middy object
    
    Returns
    -------
    dict
    """

    if midi_obj.time_signature_changes:
        # if there was change, take the first beat time from the earliest signature
        midi_obj.time_signature_changes.sort(key=lambda x: x.time)
        first_beat_time = midi_obj.time_signature_changes[0].time
    else:
        first_beat_time = midi_obj.estimate_beat_start()

    tc_times, tempi = midi_obj.get_tempo_changes()

    if len(midi_obj.time_signature_changes) == 1:
        time_sign = "{}/{}".format(
            midi_obj.time_signature_changes[0].numerator,
            midi_obj.time_signature_changes[0].denominator,
        )
    else:
        time_sign = None

    midi_info = {
        "first_beat_time": first_beat_time,
        "num_time_signature_change": len(midi_obj.time_signature_changes),
        "time_signature": time_sign,
        "tempo": tempi[0] if len(tc_times) == 1 else None,
    }

    return midi_info

def get_merged(multitrack):
    """Return a `pypianoroll.Multitrack` instance with piano-rolls merged to
    five tracks (Bass, Drums, Guitar, Piano and Strings)
    
    Parameters
    ----------
    multitrack : pypianoroll.Multitrack

    Returns
    -------
    pypianoroll.Multitrack
    """
    category_list = {"Bass": [], "Drums": [], "Guitar": [], "Piano": [], "Strings": []}
    program_dict = {"Piano": 0, "Drums": 0, "Guitar": 24, "Bass": 32, "Strings": 48}

    for idx, track in enumerate(multitrack.tracks):
        if track.is_drum:
            category_list["Drums"].append(idx)
        elif track.program // 8 == 0:
            category_list["Piano"].append(idx)
        elif track.program // 8 == 3:
            category_list["Guitar"].append(idx)
        elif track.program // 8 == 4:
            category_list["Bass"].append(idx)
        else:
            category_list["Strings"].append(idx)

    tracks = []
    for key in category_list:
        if category_list[key]:
            pianoroll = Multitrack(
                tracks=[multitrack[i] for i in category_list[key]]
            ).blend()
        else:
            pianoroll = None
        tracks.append(
            Track(
                pianoroll=pianoroll,
                program=program_dict[key],
                is_drum=key == "Drums",
                name=key,
            )
        )
    return Multitrack(
        tracks=tracks,
        tempo=multitrack.tempo,
        downbeat=multitrack.downbeat,
        resolution=multitrack.resolution,
        name=multitrack.name,
    )

def create_multitracks(filepath, multitrack_path):
    """Save a multi-track piano-roll converted from a MIDI file to target
    dataset directory and update MIDI information to `midi_dict`
    
    Parameters
    ----------
    filepath : str
        Path to midi file to be converted.
    multitrack_path : str
        Path to save outputs as .npz files.
    
    Returns
    -------
    List[str, dict]
    """
    midi_name = os.path.splitext(os.path.basename(filepath))[0]

    try:
        # Create the multitrack object
        multitrack = Multitrack(resolution=24, name=midi_name)

        pm = pretty_midi.PrettyMIDI(filepath)
        midi_info = get_midi_info(pm)

        # This adds the pianoroll to the multitrack
        multitrack = from_pretty_midi(pm)
        merged = get_merged(multitrack)

        make_sure_path_exists(multitrack_path)
        merged.save(os.path.join(multitrack_path, midi_name + ".npz"))

        return [midi_name, midi_info]

    except TypeError:
        print(f"There was a type error when loading {midi_name}")
        return None

def midi_filter(midi_info, beat_time=0, time_signature_changes=0, allowed_signatures=["4/4"]):
    """Return True for qualified midi files and False for unwanted ones.
    
    Files are qualified based on `first_beat_time`, `num_time_signature_change` and
    `time_signature`.
    If you do not wish to filter for time signatures pass None to `allowed_signatures`.

    Returns
    -------
    boolean
    """
    if midi_info["first_beat_time"] > beat_time:
        return False
    elif midi_info["num_time_signature_change"] > time_signature_changes:
        return False
    elif allowed_signatures:
        if midi_info["time_signature"] not in allowed_signatures:
            return False
    return True

# step 2
def convert_and_clean_midis(
    root_path,
    cleaned_fpath="test/cleaned",
    multitrack_path="test/multitrack",
    test_fpath="test/origin_midi",
    **filter_kwargs
):
    """Loads the midis selected in the first step and converts them to pypianoroll.Multitrack and filters
    based on the given rules.
    
    Creating the files as multitrack also merges instruments into a single track.

    Parameters
    ----------
    root_path : str
    cleaned_fpath : str
    multitrack_path : st
    test_fpath : str
    filter_kwargs : dict
    """
    
    # First step: load midis, create multitracks of each and save them
    midi_paths = get_midi_path(os.path.join(ROOT_PATH, "test/origin_midi"))
    logging.info(f"Found {len(midi_paths)} midi files")

    track_metadata = {}
    midi_tracks = [create_multitracks(midi_path, multitrack_path) for midi_path in midi_paths]
    for (name, track) in midi_tracks.items():
        if name is not None:
            track_metadata[name] = track

    with open(os.path.join(ROOT_PATH, "test/midis.json"), "w") as outfile:
        json.dump(track_metadata, outfile)
    logging.info("[Done] {} files out of {} have been successfully converted".format(len(track_metadata), len(midi_paths)))


    # Second step: open the files filter them based on track information such as beat time, time signature changes..
    with open(os.path.join(ROOT_PATH, 'test/midis.json')) as infile:
        track_metadata = json.load(infile)
    count = 0
    make_sure_path_exists(cleaned_fpath)
    clean_metadata = {}
    for key in track_metadata:
        if midi_filter(track_metadata[key], **filter_kwargs):
            clean_metadata[key] = track_metadata[key]
            count += 1
            shutil.copyfile(os.path.join(multitrack_path, key + '.npz'),
                            os.path.join(cleaned_fpath, key + '.npz'))

    with open(os.path.join(ROOT_PATH, 'test/midis_clean.json'), 'w') as outfile:
        json.dump(clean_metadata, outfile)
    logging.info("[Done] {} files out of {} have been successfully cleaned".format(count, len(track_metadata)))


# Step 1
def train_test_split(root_path, genre, test_ratio=0.1):
    """Splits the files in a given directory into training and test sets
    
    Parameters
    ----------
    root_path : str
    genre : str
    test_ratio:
        Ratio of files used for testing the model.
    """
    filenames = [f for f in os.listdir(os.path.join(root_path, genre))]
    test_fpath = os.path.join(root_path, "test", "origin_midi")

    idx = np.random.choice(len(filenames), int(test_ratio * len(filenames)), replace=False)
    for i in idx:
        shutil.move(
            os.path.join(root_path, genre, filenames[i]),
            os.path.join(test_fpath, filenames[i]),
        )
    logger.info(len(idx), f"files saved for test in {test_fpath}")

def main(argv):
    train_test_split(...)
    convert_and_clean_midis(...)
    

if __name__ == "__main__":
    main(sys.argv[1:])