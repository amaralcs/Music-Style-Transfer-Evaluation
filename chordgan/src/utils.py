import numpy as np

def restore_pianoroll(piano_roll, low_note=24, high_note=102):
    """Restores a trimmed piano roll matrix to the original size.

    The piano roll is restored to size (127, n_timesteps) by appending a matrix
    of zeros to the start and to the end of the piano roll.

    Parameters
    ----------
    piano_roll : np.array
        The piano roll to restore.
    low_note : int
        Index of lowest note to keep.
    high_note : int
        Index of highest note to keep.
    """
    n_timesteps = piano_roll.shape[1]

    # Size of arrays to use as padding
    pad_low = low_note - 0
    pad_high = 127 - high_note

    low_block = np.zeros((pad_low, n_timesteps))
    high_block = np.zeros((pad_high, n_timesteps))

    return np.r_[low_block, piano_roll, high_block]