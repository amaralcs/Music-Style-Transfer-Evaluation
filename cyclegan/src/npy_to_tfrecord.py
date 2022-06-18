import os
from glob import glob
import numpy as np

import tensorflow as tf
from tensorflow.train import Feature, Features, Example, BytesList


def np_to_tfrecord(array):
    feature = {"array": Feature(bytes_list=BytesList(value=[array.numpy()]))}
    features = Features(feature=feature)
    return Example(features=features)


if __name__ == "__main__":
    """This should be implemented in the prepare data step, rather than writing to a numpy file,
    We can write directly to tf record
    """
    path = "data/JC_C"
    outpath = "data/JC_C/tfrecord"
    set_type = "train"
    os.makedirs(os.path.join(outpath, set_type), exist_ok=True)

    fnames = glob(os.path.join(path, set_type, "*.*"))

    for full_path in fnames:
        array = np.load(full_path).astype(np.float32)
        serialized_array = tf.io.serialize_tensor(array)
        record = np_to_tfrecord(serialized_array)

        fname = os.path.split(full_path)[-1].replace(".npy", ".tfrecord")
        with tf.io.TFRecordWriter(os.path.join(outpath, set_type, fname)) as writer:
            writer.write(record.SerializeToString())
