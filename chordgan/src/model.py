import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam

from src.utils import restore_pianoroll
from reverse_pianoroll import piano_roll_to_pretty_midi



def xavier_init(size, dtype=None):
    input_dim = size[0]
    xavier_stddev = 1.0 / tf.sqrt(input_dim / 2)
    return tf.random.normal(shape=size, stddev=xavier_stddev)


class ChordGAN(Model):
    def __init__(
        self,
        note_range=78,
        chroma_dims=12,
        n_timesteps=4,
        generator_units=128,
        discriminator_units=512,
        lambda_=100,
        loss_func=MeanSquaredError(),
        name="ChordGAN",
        **kwargs,
    ):
        """Instantiates the ChordGAN class.

        Parameters
        ----------
        note_range : int
            Range of lowest/highest note on the piano roll
        chroma_dims : int
        n_timesteps : int
            This is the number of timesteps that we will create at a time
        generator_units : int
        discriminator_units : int
        lambda_ : int
            The value of lambda_, a parameter controling ____ TODO: fill in
        loss_func : Instance of tf.keras.losses
            The type of loss to use for the generated fake samples.
        name : str
            The name of the model.
        **kwargs :
            Other parameters to be passed to tf.keras.models.Model
        """
        super(ChordGAN, self).__init__(name=name, **kwargs)
        self.note_range = note_range
        self.chroma_dims = chroma_dims
        self.n_timesteps = n_timesteps
        self.generator_units = generator_units
        self.discriminator_units = discriminator_units
        self.lambda_ = lambda_
        self.loss_func = loss_func

        self.X_dim = (
            self.note_range
        )  # This is the size of the visible layer.
        self.Z_dim = self.chroma_dims

        self.generator = self._build_generator(self.X_dim, self.Z_dim)
        self.discriminator = self._build_discriminator(self.X_dim, self.Z_dim)

        # used in the losses
        self.binary_cross_entropy = BinaryCrossentropy(from_logits=True)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "note_range": self.note_range,
            "chroma_dims": self.chroma_dims,
            "n_timesteps": self.n_timesteps,
            "generator_units": self.generator_units,
            "discriminator_units": self.discriminator_units,
            "lambda_": self.lambda_,
            "loss_func": self.loss_func,
        }

    def _build_generator(self, X_dim, Z_dim):
        """Builds the generator using the Keras functional API.

        Parameters
        ----------
        X_dim : int
            The number of dimensions of the song input.
        Z_dim : int
            The number of dimensions of the chromagram input.

        Returns
        -------
        tensorflow.keras.models.Model
            The generator model.
        """
        inputs = Input(shape=(None, Z_dim), name="g_input")
        z = Dense(
            self.generator_units,
            kernel_initializer=xavier_init,
            activation="relu",
        )(inputs)
        output = Dense(X_dim, kernel_initializer=xavier_init, activation="sigmoid")(z)

        generator = Model(inputs=[inputs], outputs=[output], name="generator")
        generator.add_loss(self.generator_loss)
        return generator

    def generator_loss(self, y_true, y_preds):
        """
        Note: While the paper describes L1 loss (which is MAE) the code uses MSE.

        Parameters
        ----------
        y_true:
            The real data samples.
        y_preds : tuple
            Containing the following
                D_fake_logits:
                    The logits output by the discriminator given the fake sample.
                fake_samples :
                    The sample data created by the generator.
        lambda_ : float, Optional
            Normalization factor
        loss_func :
            The type of loss to use for the generated fake samples.
        """
        true_samples = y_true
        D_fake_logits, fake_samples = y_preds

        # Probability the generator fooled the discriminator (i.e. all predictions on fake samples were labelled 1)
        G_fooling = tf.reduce_mean(
            self.binary_cross_entropy(tf.ones_like(D_fake_logits), D_fake_logits)
        )
        G_loss = tf.reduce_mean(self.loss_func(true_samples, fake_samples))
        return G_fooling + self.lambda_ * G_loss

    def _build_discriminator(self, X_dim, Z_dim):
        """Builds the discriminator using the Keras functional API.

        Parameters
        ----------
        X_dim : int
            The number of dimensions of the song input.
        Z_dim : int
            The number of dimensions of the chromagram input.

        Returns
        -------
        tensorflow.keras.models.Model
            The discriminator model.
        """
        data_inputs = Input(shape=(None, X_dim), name="data_input")
        chroma_inputs = Input(shape=(None, Z_dim), name="chroma_input")

        x = Concatenate(axis=2)([data_inputs, chroma_inputs])
        x = Dense(
            self.discriminator_units,
            kernel_initializer=xavier_init,
            activation="relu",
        )(x)
        logits = Dense(1, kernel_initializer=xavier_init)(x)
        probas = sigmoid(logits)

        discriminator = Model(
            inputs=[data_inputs, chroma_inputs],
            outputs=[logits, probas],
            name="discriminator",
        )
        discriminator.add_loss(self.discriminator_loss)
        return discriminator

    def discriminator_loss(self, y_true, y_preds):
        """TODO: docstring"""
        D_true_logits = y_true
        D_fake_logits = y_preds

        # Discriminator should identify the true samples as 1s
        D_true_loss = tf.reduce_mean(
            self.binary_cross_entropy(tf.ones_like(D_true_logits), D_true_logits)
        )
        # And the fake samples as 0s
        D_fake_loss = tf.reduce_mean(
            self.binary_cross_entropy(tf.zeros_like(D_fake_logits), D_fake_logits)
        )
        return D_true_loss + D_fake_loss

    def compile(self, d_optimizer=Adam(), g_optimizer=Adam()):
        """Compiles the model given optimizers for the discriminator and generator.

        Parameters
        ----------
        d_optimizer : tf.keras.optimizer
        g_optimizer : tf.keras.optimizer
        """
        super(ChordGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def train_step(self, inputs):
        actual_song, chroma = inputs

        # train the discriminator
        with tf.GradientTape() as d_tape:
            fake_song = self.generator(chroma, training=False)
            d_true_logits, _ = self.discriminator([actual_song, chroma], training=True)
            d_fake_logits, _ = self.discriminator([fake_song, chroma], training=True)

            d_loss = self.discriminator_loss(d_true_logits, d_fake_logits)

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_weights)
        )

        # train the generator
        with tf.GradientTape() as g_tape:
            fake_song = self.generator(chroma, training=True)

            d_true_logits, _ = self.discriminator([actual_song, chroma], training=False)
            d_fake_logits, _ = self.discriminator([fake_song, chroma], training=False)
            g_loss = self.generator_loss(actual_song, (d_fake_logits, fake_song))

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))

        return {"d_loss": d_loss, "g_loss": g_loss}

    def call(self, chroma, low_note=24, high_note=102):
        """Converts a song given its chroma. The chroma needs to have been correctly reshaped
        as per the preprocessing function.

        Parameters
        ----------
        chroma : np.array
            Chromagram of the song to convert.
        low_note : int
            Index of lowest note to keep.
        high_note : int
            Index of highest note to keep.
        """
        # The generator returns a tensor of shape
        converted_song = self.generator(chroma).numpy()[0]

        piano_roll = restore_pianoroll(converted_song.T, low_note, high_note)
        piano_roll_thresh = (piano_roll >= 0.5) * 127 # set all non-zero velocities to 127

        return piano_roll_to_pretty_midi(piano_roll_thresh, fs=16)

    def summary(self):
        self.generator.summary()
        self.discriminator.summary()
