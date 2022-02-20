import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError


def xavier_init(size, dtype=None):
    input_dim = size[0]
    xavier_stddev = 1.0 / tf.sqrt(input_dim / 2)
    return tf.random.normal(shape=size, stddev=xavier_stddev)


class ChordGANGenerator(Model):
    def __init__(
        self,
        X_dim,
        Z_dim,
        generator_units=128,
        lambda_=100,
        loss_func=MeanSquaredError(),
        name="ChordGANGenerator",
        **kwargs,
    ):
        super(ChordGANGenerator, self).__init__(name=name, **kwargs)
        self.generator_units = generator_units
        self.lambda_ = lambda_
        self.loss_func = loss_func
        self.generator = self._build_generator(X_dim, Z_dim)

    def _build_generator(self, X_dim, Z_dim):
        inputs = Input(shape=[None, Z_dim], name="g_input")
        z = Dense(
            self.generator_units,
            kernel_initializer=xavier_init,
            activation="relu",
        )(inputs)
        output = Dense(X_dim, kernel_initializer=xavier_init, activation="sigmoid")(z)

        generator = Model(inputs=[inputs], outputs=[output], name="generator")
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

        binary_cross_entropy = BinaryCrossentropy(from_logits=True)

        # Probability the generator fooled the discriminator (i.e. all predictions on fake samples were labelled 1)
        G_fooling = tf.reduce_mean(
            binary_cross_entropy(tf.ones_like(D_fake_logits), D_fake_logits)
        )
        G_loss = tf.reduce_mean(self.loss_func(true_samples, fake_samples))
        return G_fooling + self.lambda_ * G_loss

    def call(self, inputs):
        print("gen input shape", inputs.shape)
        self.generator.add_loss(self.generator_loss)
        return self.generator(inputs)

    def summary(self):
        self.generator.summary()


class ChordGANDiscriminator(Model):
    def __init__(
        self,
        X_dim,
        Z_dim,
        generator_units=128,
        discriminator_units=512,
        lambda_=100,
        loss_func=MeanSquaredError(),
        name="ChordGANDiscriminator",
        **kwargs,
    ):
        super(ChordGANDiscriminator, self).__init__(name=name, **kwargs)
        self.generator_units = generator_units
        self.discriminator_units = discriminator_units
        self.lambda_ = lambda_
        self.loss_func = loss_func

        self.discriminator = self._build_discriminator(X_dim, Z_dim)

    def _build_discriminator(self, X_dim, Z_dim):
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

    def discriminator_loss(y_true, y_preds):
        D_true_logits = y_true
        D_fake_logits = y_preds
        binary_cross_entropy = BinaryCrossentropy(from_logits=True)

        # Discriminator should identify the true samples as 1s
        D_true_loss = tf.reduce_mean(
            binary_cross_entropy(tf.ones_like(D_true_logits), D_true_logits)
        )
        # And the fake samples as 0s
        D_fake_loss = tf.reduce_mean(
            binary_cross_entropy(tf.zeros_like(D_fake_logits), D_fake_logits)
        )
        return D_true_loss + D_fake_loss

    def call(self, inputs):
        self.discriminator.add_loss(self.discriminator_loss)
        return self.discriminator(inputs)

    def summary(self):
        self.discriminator.summary()
