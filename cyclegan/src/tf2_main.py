import sys
import argparse
from yaml import safe_load, YAMLError
import os

from tf2_model import CycleGAN
from tf2_classifier import Classifier


def parse_args(argv):
    """Parses input options for this module.

    Parameters
    ----------
    argv : List
        List of input parameters to this module

    Returns
    -------
    Argparser
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", default="config.yaml")
    parser.add_argument("--phase", dest="phase", default="train", help="train, test")
    parser.add_argument(
        "--log_dir", dest="log_dir", default="./log", help="logs are saved here"
    )
    parser.add_argument(
        "--type", dest="type", default="cyclegan", help="cyclegan or classifier"
    )
    return parser.parse_args()


def setup_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def load_config(config_path):
    with open(config_path, "r") as config_file:
        try:
            model_config = safe_load(config_file)
        except YAMLError as e:
            raise e
    return model_config["CycleGAN"]

def main(argv):
    args = parse_args(argv)
    type = args.type
    phase = args.phase
    config_path = args.config_path
    model_config = load_config(config_path)

    if type == "cyclegan":
        model = CycleGAN(**model_config)
        model.train() if phase == "train" else model.test()

    # if type == "classifier":
    #     classifier = Classifier()
    #     classifier.train() if phase == "train" else classifier.test()


if __name__ == "__main__":
    main(sys.argv[1:])