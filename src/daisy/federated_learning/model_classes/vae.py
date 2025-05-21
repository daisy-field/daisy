# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""A variational autoencoder class for jamming detection.

Author: Simon Torka
Modified: 22.01.25
"""

import os
import json
import pickle

import tensorflow as tf
from tensorflow import keras
from keras.layers import concatenate


class DetectorVAE:
    """A Variational Autoencoder (VAE) designed for anomaly detection tasks.

    This implementation includes optional U-Net-style skip connections and allows
    for flexible architecture definitions for the encoder and decoder networks.
    """

    def __init__(self, input_dim, hidden_layers, latent_dim, unet_switch=False):
        """Initializes the VAE with specified parameters.

        :param input_dim: Dimensionality of the input data.
        :param hidden_layers: List of integers defining the sizes of hidden layers.
        :param latent_dim: Dimensionality of the latent space.
        :param unet_switch: Whether to enable U-Net-style skip connections.
        """
        self.is_unet = unet_switch

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.latent_dim = latent_dim

        self.encoder_layers = []

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.model, self.loss = self.build_vae()

    def build_encoder(self):
        """Builds the encoder part of the VAE.

        :return: A Keras Model representing the encoder.
        """
        # Encoder
        inputs = keras.layers.Input(shape=(self.input_dim,))
        h = inputs

        # Dynamisch versteckte Schichten im Encoder hinzufügen
        for idx, units in enumerate(self.hidden_layers):
            if idx == 0:
                h = keras.layers.Dense(
                    units,
                    activation=None,
                    kernel_initializer=keras.initializers.HeNormal(),
                    activity_regularizer=keras.regularizers.l1_l2(l1=0.0001, l2=0.0005),
                )(h)
                # LeakyReLU als eigene Schicht
                h = keras.layers.LeakyReLU(alpha=0.01)(h)
            else:
                h = keras.layers.Dense(
                    units,
                    activation="relu",
                    kernel_initializer=keras.initializers.HeNormal(),
                    activity_regularizer=keras.regularizers.l2(1e-4),
                )(h)
                # LeakyReLU als eigene Schicht
                # h = keras.layers.LeakyReLU(alpha=0.01)(h)

            # h = keras.layers.Dropout(0.2)(h)

            self.encoder_layers.append(h)

        z_mean = keras.layers.Dense(self.latent_dim, name="z_mean")(h)
        z_log_var = keras.layers.Dense(self.latent_dim, name="z_log_var")(h)
        z = Sampling()([z_mean, z_log_var])

        return keras.models.Model(inputs, [z_mean, z_log_var, z], name="encoder")

    def build_decoder(self):
        """Builds the decoder part of the VAE.

        :return: A Keras Model representing the decoder.
        """
        # Decoder
        latent_inputs = keras.layers.Input(shape=(self.latent_dim,))
        h = latent_inputs

        # Dynamisch versteckte Schichten im Decoder hinzufügen
        for idx, units in enumerate(reversed(self.hidden_layers)):
            h = keras.layers.Dense(
                units,
                activation="relu",
                activity_regularizer=keras.regularizers.l2(1e-4),
            )(h)
            # LeakyReLU als eigene Schicht
            # h = keras.layers.LeakyReLU(alpha=0.01)(h)

            # h = keras.layers.Dropout(0.2)(h)

            # Skip-Connection => only for unet
            if self.is_unet:
                h = concatenate([h, self.encoder_layers[-(idx + 1)]])

        h = keras.layers.Dense(self.input_dim, activation="sigmoid")(h)

        return keras.models.Model(latent_inputs, h, name="decoder")

    def build_vae(self):
        """Builds the complete VAE by combining encoder and decoder.

        :return: A tuple of the VAE model and its loss function.
        """
        inputs = keras.layers.Input(shape=(self.input_dim,))
        z_mean, z_log_var, z = self.encoder(inputs)
        outputs = self.decoder(z)

        model = keras.models.Model(inputs, outputs, name="vae")
        loss = DetectorVAE.vae_loss(inputs, outputs, z_log_var, z_mean, self.input_dim)
        return model, loss

    @staticmethod
    # Eigenständige Schicht für den Verlust
    def vae_loss(x, x_decoded_mean, z_log_var, z_mean, input_dim):
        """Defines the loss function for the VAE.

        Combines reconstruction loss and KL divergence.

        :param x: Original input data.
        :param x_decoded_mean: Reconstructed data from the decoder.
        :param z_log_var: Log variance of the latent distribution.
        :param z_mean: Mean of the latent distribution.
        :return: The calculated loss value.
        """
        xent_loss = tf.keras.losses.mean_squared_error(
            x, x_decoded_mean
        )  # binary_crossentropy
        xent_loss *= input_dim

        # xent_loss *= input_dim
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        total_loss = tf.reduce_mean(xent_loss + kl_loss)
        return total_loss

    # Custom Sampling Layer

    def save(self, path: str):
        """Speichert das Modell und die Konfiguration."""
        os.makedirs(path, exist_ok=True)

        # Modell speichern
        model_path = os.path.join(path, "model")
        self.model.save(model_path)

        # Konfiguration speichern
        config = {
            "input_dim": self.input_dim,
            "hidden_layers": self.hidden_layers,
            "latent_dim": self.latent_dim,
            "unet_switch": self.is_unet,
        }
        config_path = os.path.join(path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        """Speichert eine Loss-Funktion mit Pickle."""
        loss_path = os.path.join(path, "loss_function.pkl")
        with open(loss_path, "wb") as f:
            pickle.dump(self.loss, f)

    @classmethod
    def load(cls, path: str):
        """Lädt das Modell und die Konfiguration."""
        # Konfiguration laden
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # VAE-Wrapper neu instanziieren
        vae = cls(**config)

        # Modell laden (custom_objects wegen Sampling!)
        model_path = os.path.join(path, "model")
        vae.model = keras.models.load_model(
            model_path, custom_objects={"Sampling": Sampling}
        )

        """Lädt eine gespeicherte Loss-Funktion."""
        loss_path = os.path.join(path, "loss_function.pkl")
        with open(loss_path, "rb") as f:
            loss = pickle.load(f)

        return vae, loss


class Sampling(keras.layers.Layer):
    """Custom Keras layer for sampling in the VAE.

    Implements the reparameterization trick to sample from the latent space.
    """

    def call(self, inputs):
        """Samples a latent vector using the reparameterization trick.

        :param inputs: A tuple of mean and log variance of the latent distribution.
        :return: A sampled latent vector.
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]  # Use tf.shape instead of K.shape
        dim = tf.shape(z_mean)[1]  # Use tf.shape instead of K.shape
        epsilon = tf.random.normal(
            shape=(batch, dim)
        )  # Replace K.random_normal with tf.random.normal
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
