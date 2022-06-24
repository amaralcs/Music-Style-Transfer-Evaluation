# Initial setup to be able to load `src.cyclegan`
from imp import new_module
import os
import time
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler

from utils import load_data
from cyclegan import CycleGAN


def lr_function_wrapper(lr, epochs, step):
    """Helper function to initialize the variable_lr function.

    Parameters
    ----------
    lr : float
        The initial learning rate.
    epochs : int
        The number of epochs the model will be trained for.
    step : int
        The number of epochs to maintain the initial lr.

    Returns
    -------
    function
    """

    def variable_lr(epoch):
        """Defines a variable learning rate.

        Parameters
        ----------
        epoch : int
            The current training epoch.

        Returns
        -------
        float
            The new learning rate.
        """
        if epoch < step:
            new_lr = lr
        else:
            new_lr = lr * (epochs - epoch) / (epochs - step)

        tf.summary.scalar("learning_rate", data=new_lr, step=epoch)
        return new_lr

    return variable_lr


def get_run_logdir(root_logdir, genre_a, genre_b, epochs, batch_size):
    """Generates the paths where the logs for this run will be saved.

    Parameters
    ----------
    root_logdir : str
        The base path to use.
    genre_a : str
        The name of genre A.
    genre_b : str
        The name of genre B.
    epochs : int
        Number of epochs the model will be trained for.
    batch_size : int
        The batch size used.

    Returns
    -------
    str, str
        The full path to the logging directory as well as the name of the current run.
    """
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    model_info = "{}2{}_{}e_bs{}_{}".format(
        genre_a, genre_b, epochs, batch_size, run_id
    )
    return os.path.join(root_logdir, model_info), model_info


if __name__ == "__main__":
    # Set training configuration, load data
    # path_a = "data/JC_C_cp/tfrecord" # dummy dir with less data
    path_a = "data/JC_C/tfrecord"
    # path_b = "data/JC_J_cp/tfrecord"
    path_b = "data/JC_J/tfrecord"
    genre_a = "classic"
    genre_b = "jazz"

    model_output = "trained_models"
    os.makedirs(model_output, exist_ok=True)
    log_dir = "train_logs"
    os.makedirs(log_dir, exist_ok=True)

    lr = 0.0002
    beta_1 = 0.5
    optimizer_params = dict(learning_rate=lr, beta_1=beta_1)
    batch_size = 32
    epochs = 15
    step = 3

    dataset = load_data(
        path_a, path_b, "train", batch_size=batch_size, cycle_length=500
    )

    # print("showing example")
    # for (a, b) in dataset.take(1):
    #     print(a.shape, b.shape)

    # Setup monitoring and callbacks
    run_logdir, model_info = get_run_logdir(
        log_dir, genre_a, genre_b, epochs, batch_size
    )
    file_writer = tf.summary.create_file_writer(run_logdir + "/metrics")
    file_writer.set_as_default()
    lr_function = lr_function_wrapper(lr, epochs, step)
    callbacks = [LearningRateScheduler(lr_function), TensorBoard(log_dir=run_logdir)]
    callbacks = None

    # Setup model
    model = CycleGAN(genre_a, genre_b)
    model.build_model(default_init=optimizer_params)

    model.fit(dataset, epochs=epochs, callbacks=callbacks)

    model.save_weights(f"{model_output}/{model_info}/weights/")

    new_model = CycleGAN(genre_a, genre_b)
    new_model.load_weights(f"{model_output}/{model_info}/weights/")

    dummy_out = new_model(list(dataset.take(1))[0])
    print("Dummy output shape", dummy_out.shape)
