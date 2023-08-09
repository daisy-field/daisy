"""
    Federated autoencoder that implements federated_model interface.

    Author: Seraphin Zunzer
    Modified: 09.08.23
"""


import keras
import tensorflow as tf
from federated_model.federated_model import FederatedModel

input_size = 70


class FedAutoencoder(FederatedModel):
    """Class for federated autoencoder"""

    def __init__(self):
        """
        Build the autoencoder

        :return: built model
        """
        encoder = tf.keras.models.Sequential([
            keras.layers.Dense(input_size, input_shape=(input_size,)),
            keras.layers.Dense(35),
            keras.layers.Dense(18),
        ])
        decoder = tf.keras.models.Sequential([
            keras.layers.Dense(35, input_shape=(18,)),
            keras.layers.Dense(input_size),
            keras.layers.Activation("sigmoid"),
        ])
        input_format = keras.layers.Input(shape=(input_size,))
        self.model = tf.keras.models.Model(inputs=input_format, outputs=decoder(encoder(input_format)))

    def compile_model(self):
        """
        Compile the model for prediction

        :return: compiled model
        """
        self.model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='mse', metrics=[])
        return self.model

    def set_model_weights(self, weights):
        self.model.set_weights(weights)

    def get_model_weights(self):
        self.model.get_weights()

    def fit_model(self, **kwargs):
        self.model.fit(**kwargs)

    def model_predict(self, **kwargs):
        self.model.predict(**kwargs)




