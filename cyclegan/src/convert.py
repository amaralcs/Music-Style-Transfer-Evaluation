# Initial setup to be able to load `src.cyclegan`
import os
import numpy as np
import logging

from utils import load_data
from tf2_utils import save_midis
import write_midi
from cyclegan import CycleGAN

logger = logging.getLogger("convert_logger")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    # Set training configuration, load data
    path_a = "data/JC_C_cp/tfrecord"  # dummy dir with less data
    # path_a = "data/JC_C/tfrecord"
    path_b = "data/JC_J_cp/tfrecord"
    # path_b = "data/JC_J/tfrecord"
    genre_a = "classic"
    genre_b = "jazz"

    model_name = "classic2jazz_15e_bs32_run_2022_06_22-20_08_16"
    model_fpath = os.path.join(os.getcwd(), "trained_models", model_name, "weights", "")

    outpath = f"converted/{model_name}"
    os.makedirs(outpath, exist_ok=True)

    dataset = load_data(path_a, path_b, "test", shuffle=False)

    # Setup model
    logger.debug(f"Loading model from {model_fpath}")
    model = CycleGAN(genre_a, genre_b)
    model.load_weights(model_fpath)
    logger.debug(f"\tsuccess!")

    logger.debug(f"Converting and saving results to {outpath}")
    for batch in dataset:
        original_inputs, converted, cycled = model(batch)

        for idx, (original, transfer, cycle) in enumerate(
            zip(original_inputs, converted, cycled)
        ):
            save_midis(original[np.newaxis, ...], f"{outpath}/{idx}_original.mid")
            save_midis(transfer[np.newaxis, ...], f"{outpath}/{idx}_transfer.mid")
            save_midis(cycle[np.newaxis, ...], f"{outpath}/{idx}_cycle.mid")
