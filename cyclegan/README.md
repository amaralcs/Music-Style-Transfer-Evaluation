# CycleGAN-Music-Style-Transfer-Refactorization

The contents of this folder were cloned from the following Git repo:
https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer-Refactorization


# Usage


## Preparing the data
To prep the data, place a folder in this directory and create subdirectories labelled according to the genre of of music present in each subdirectory. Example:
```
datasets
    |
    |-- pop
    |   |-- track1.midi
    |   |-- track2.midi
    |   |-- ...
    |
    |-- jazz
        |-- track1.midi
        |-- track2.midi
        |-- ...
```
Then run `src/prepare_data.py`. The basic usage will be to pass your root directory (datasets in the example above) along with the genre subdirectory. For example

```
python prepare_data.py datasets pop
```

The resulting train and test files will be saved under `datasets/phrases`. 

## Training

Change the options as needed in `src/config.yaml` and you're good to go! To run the script use: 
```
python tf2_main.py
```

Note: The default paths is set to load the data and save results/logs/checkpoints in the parent directory of `src`, hence it assumes that `tf2_main.py` is being invoked from within `src`.

## Converting new samples

TBD
