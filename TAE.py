
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

class TAE:
    def __init__(self, input_shape, output_shape, latent_space_dim):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.latent_space_dim = latent_space_dim

        self.encoder = None
        self.decoder = None
        self.model = None

        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def compile(self, learning_rate=0.001):
        opt = keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        mse_loss = keras.losses.MeanSquaredError()
        self.model.compile(optimizer=opt, loss=mse_loss)

    def train(self, x_sparse, x_train, batch_size, num_epochs):
        self.model.fit(x_sparse, x_train, batch_size=batch_size,
                       epochs=num_epochs, shuffle=True, validation_split=0.1)

    def predict(self, data):
        latent_representation = self.encoder.predict(data)
        reconstructed_data = self.decoder.predict(latent_representation)
        return reconstructed_data, latent_representation

    def _build_encoder(self):
        encoder_input = keras.layers.Input(shape=self.input_shape, name="encoder_input")

        x = keras.layers.Flatten()(encoder_input)
        x = keras.layers.Dense(64, activation="elu")(x)
        encoder_output = keras.layers.Dense(self.latent_space_dim, activation="elu")(x)

        self.encoder = keras.Model(encoder_input, encoder_output, name="encoder")

    def _build_decoder(self):
        decoder_input = keras.layers.Input(shape=self.latent_space_dim, name="decoder_input")

        x = keras.layers.Dense(64, activation="elu")(decoder_input)
        # x = keras.layers.Dense(64, activation="elu")(x)
        x = keras.layers.Dense(np.prod(self.output_shape), activation="linear")(x)
        decoder_output = keras.layers.Reshape(self.output_shape, name="decoder_output")(x)

        self.decoder = keras.Model(decoder_input, decoder_output, name="decoder")

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = keras.Model(model_input, model_output, name="autoencoder")
