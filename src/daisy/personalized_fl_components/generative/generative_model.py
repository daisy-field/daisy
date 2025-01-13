# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""An adapted federated node implementing a personalization algorithm, using generative learning
for learning personalized models. Instead of handling the different shapes and sizes of models in the model aggreagtor,
this approach aims to train equal Generative Adversarial Networks (GANs) on each node to produce synthetic data with
similar statistics to the local data stream. These data generators are exchanged and aggregated using the classical
federated learning approach e.g. FedAvg and used by the nodes to mix synthetic data into the arriving data stream.

Author: Seraphin Zunzer
Modified: 13.01.25
"""

from abc import abstractmethod
from typing import Self
import logging

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow import keras
from tensorflow.keras import layers

from daisy.federated_learning import FederatedModel


class GenerativeModel(FederatedModel):
    """An abstract Generative Model wrapper that offers the same methods, no matter the type of
    underlying model. Must always be implemented if a new model type is to be used in
    the personalized federated learning system using generative models.
    """

    @abstractmethod
    def create_synthetic_data(self, n) -> Tensor:
        """Generates n number of augmented data samples

        :param n: Number of datapoints to generate.
        :return: Predicted output tensor.
        """
        raise NotImplementedError


class GenerativeGAN(GenerativeModel):
    """Implementation of the Tensorflow DCGAN Tutorial with adapted model layers.
    See: https://www.tensorflow.org/tutorials/generative/dcgan
    """

    _generator: keras.Model
    _discriminator: keras.Model

    def __init__(
        self,
        generator_optimizer: str | keras.optimizers.Optimizer,
        discriminator_optimizer: str | keras.optimizers.Optimizer,
        epochs: int = 1,
        input_size: int = 100,
        noise_dim: int = 100,
    ):
        """Creates a new tensorflow federated model from a given model. This also
        compiles the given model, requiring a set of additional arguments
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile).

        :param model: Underlying model to be wrapped around.
        :param epochs: Number of epochs (rounds) during training.
        """
        self._generator_optimizer = generator_optimizer
        self._discriminator_optimizer = discriminator_optimizer
        self._logger = logging.getLogger("GenerativeGAN")

        self._epochs = epochs
        self._input_size = input_size
        self._noise_dim = noise_dim

    def set_parameters(self, parameters: list[np.ndarray]):
        """Updates the internal parameters of the generator of the generative model.

        :param parameters: Parameters to update the generator with.
        """
        self._logger.debug("Set generator weights")
        self._generator.set_weights(parameters)
        self._logger.debug("Set successful")
        return

    def get_parameters(self) -> list[np.ndarray]:
        """Retrieves the weights of the underlying GAN generator model.

        :return: Weights of the model.
        """
        self._logger.info("Get generator weights")
        params = self._generator.get_weights()
        return params

    def make_generator_model(self):
        """Create generator model

        :return: generator model
        """
        model = tf.keras.Sequential()
        model.add(layers.Dense(512, input_dim=self._input_size, name="GeneratorInput"))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(256))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(128))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dense(self._input_size, name="GeneratorOutput"))
        self._logger.info("Generator created...")
        return model

    def make_discriminator_model(self):
        """Create discriminator model

        :return: discriminator model
        """

        model = tf.keras.Sequential()
        model.add(
            layers.Dense(128, input_dim=self._input_size, name="DiscriminatorInput")
        )
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(256))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(128))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation="sigmoid", name="DiscriminatorOutput"))
        self._logger.debug("Discriminator created...")
        return model

    @classmethod
    def create_gan(
        cls,
        input_size: int,
        generator_optimizer: str | keras.optimizers.Optimizer = "Adam",
        discriminator_optimizer: str | keras.optimizers.Optimizer = "Adam",
        epochs: int = 10,
    ) -> Self:
        """Create a generative adversarial network and save it locally.
        It will be used to create synthetic data of the data present at the node.
        The GAN's model weights themselves can be shared with the aggregation server
        and aggregated using traditional aggregation methods like FedAvg.
        """

        gan = GenerativeGAN(
            generator_optimizer,
            discriminator_optimizer,
            epochs,
            input_size,
            input_size,
        )
        gan._generator = gan.make_generator_model()
        gan._discriminator = gan.make_discriminator_model()

        gan._param_split = len(gan._generator.get_weights())
        # currently not needed, can be interesting for aggregating generator AND discriminator weights

        return gan

    def train_step(self, data):
        """GAN training function.

        :param data: data to train GAN with
        """
        noise = tf.random.normal((1, self._input_size))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self._generator(noise, training=True)

            real_output = self._discriminator(data, training=True)
            fake_output = self._discriminator(generated_data, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

            self._logger.info(f"Generator loss: {gen_loss}")
            self._logger.info(f"Discriminator loss: {disc_loss}")

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self._generator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self._discriminator.trainable_variables
        )

        self._generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self._generator.trainable_variables)
        )
        self._discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self._discriminator.trainable_variables)
        )

    def generator_loss(self, fake_output):
        """Generator loss function"""
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        """Discriminator loss function"""
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def create_synthetic_data(self, n: int = 100):
        """Create n synthetic data points using the generator model.
        These datapoints should be mixed with the locally arriving
        datastream to include globally learned knowledge in the local intrusion detection model.

        :param n: number of synthetic datapoints to create
        """
        generated_data = []
        for i in range(n):
            noise = tf.random.normal([1, self._input_size])
            generated_data.append(self._generator(noise))

        generated_data_tensor = tf.concat(generated_data, axis=0)
        self._logger.info("Generated Synthetic data...")

        return generated_data_tensor

    def fit(self, dataset):
        """Start training the GAN in epochs using the train_step fuction

        :params dataset: dataset to train the GAN with.
        """
        self._logger.debug("Start GAN training...")
        for epoch in range(self._epochs):
            self.train_step(dataset)
        self._logger.debug("Finished GAN training...")

    def predict(self):
        raise NotImplementedError
