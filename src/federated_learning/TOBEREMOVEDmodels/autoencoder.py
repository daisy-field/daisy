""" FIXME MUST BE MADE COMPLIANT WITH FED MODEL ABSTRACT CLASS
    TODO CAN BE MOVED DIRECTLY INTO FEDERATED MODEL. PYTON IS NOT JAVA!
    Federated autoencoder that implements federated_models interface.

    Author: Seraphin Zunzer
    Modified: 09.08.23
"""
import logging

import keras
import tensorflow as tf

from federated_learning.federated_model import FederatedModel

input_size = 65


class FedAutoencoder(FederatedModel):
    """Class for federated autoencoder"""
    _model = None

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
        logging.info("Model created")

    def init_model(self):
        """
        Compile the model for prediction

        :return: compiled model
        """
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=[])
        logging.info("Compiled Model")
        return self.model
