# Copyright (C) 2024 DAI-Labor and others
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
Modified: 31.07.24
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
    """An abstract GAN wrapper that offers the same methods, no matter the type of
    underlying model. Must always be implemented if a new model type is to be used in
    the personalized federated learning system using generators.
    """

    @abstractmethod
    def create_synthetic_data(self, n) -> Tensor:
        """Generates n number of augmented data samples

        :param n: Number of datapoints to generate.
        :return: Predicted output tensor.
        """
        raise NotImplementedError


class GenerativeGAN(GenerativeModel):
    """ """

    _generator: keras.Model
    _discriminator: keras.Model

    def __init__(
        self,
        generator_optimizer: str | keras.optimizers.Optimizer,
        discriminator_optimizer: str | keras.optimizers.Optimizer,
        batch_size: int = 32,
        epochs: int = 1,
        input_size: int = 100,
        noise_dim: int = 10,
    ):
        """Creates a new tensorflow federated model from a given model. This also
        compiles the given model, requiring a set of additional arguments
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile).

        :param model: Underlying model to be wrapped around.
        :param batch_size: Batch size during training and prediction.
        :param epochs: Number of epochs (rounds) during training.
        """
        self._generator_optimizer = generator_optimizer
        self._discriminator_optimizer = discriminator_optimizer
        self._logger = logging.getLogger("GenerativeGAN")

        self._batch_size = batch_size
        self._epochs = epochs
        self._input_size = input_size
        self._noise_dim = noise_dim

    def set_parameters(self, parameters: list[np.ndarray]):
        """Updates the internal parameters of the model.
        %not needed
        :param parameters: Parameters to update the model with.
        """
        # TODO Seraphin: set parameters of GAN
        # check if it is sufficient to only send generator
        # self._generator.set_weights(parameters)
        # self._discriminator.set_weights(parameters)
        self._logger.info("Set generator weights")
        self._generator.set_weights(parameters)
        self._logger.info("Set successful")

        return

    def get_parameters(self) -> list[np.ndarray]:
        """Retrieves the weights of the underlying model.

        :return: Weights of the model.
        """
        # TODO Seraphin: get parameters of GAN
        # check if it is sufficient to only send generator

        self._logger.info("Get generator weights")
        params = self._generator.get_weights()
        # params.extend(self._discriminator.get_weights())
        return params

        # return self._generator.get_weights()#, self._discriminator.get_weights()]

    def make_generator_model(self):
        """Create generator model"""
        model = tf.keras.Sequential()

        # Dense layer to expand the input from latent_dim to 128 units
        model.add(layers.Dense(128, input_dim=65, name="GeneratorInput"))
        model.add(layers.LeakyReLU(alpha=0.2))

        # Dense layer to expand from 128 to 256 units
        model.add(layers.Dense(256))
        model.add(layers.LeakyReLU(alpha=0.2))

        # Dense layer to expand from 256 to 512 units
        model.add(layers.Dense(512))
        model.add(layers.LeakyReLU(alpha=0.2))

        # Final Dense layer to produce the output vector of length 65
        model.add(
            layers.Dense(self._input_size, activation="tanh", name="GeneratoOutput")
        )

        self._logger.info("Generator created...")

        return model

    def make_discriminator_model(self):
        """Create discriminator model"""
        model = tf.keras.Sequential()

        # Input layer
        model.add(layers.Dense(512, input_dim=65, name="DiscriminatorInput"))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))

        # Hidden layer
        model.add(layers.Dense(256))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))

        # Hidden layer
        model.add(layers.Dense(128))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))

        # Output layer
        model.add(layers.Dense(1, activation="sigmoid", name="DiscriminatorOutput"))
        self._logger.info("Discriminator created...")

        return model

    @classmethod
    def create_gan(
        cls,
        input_size: int,
        generator_optimizer: str | keras.optimizers.Optimizer = "Adam",
        discriminator_optimizer: str | keras.optimizers.Optimizer = "Adam",
        batch_size: int = 32,
        epochs: int = 1,
    ) -> Self:
        # TODO Seraphin: Implement creation of gan
        """Create a generative adversarial network and save it locally.
        It will be used to create synthetic data of the data present at the node.
        Based on the selected approach, the gan's model weights themselves can be shared with the aggregation server and aggregated using
        traditional aggregation methods like FedAvg (e.g. FedGen)
        or the generated data is sent to the server and mixed there (increases necessary bandwidth dramatically, e.g. PerFedGan)
        """

        gan = GenerativeGAN(
            generator_optimizer,
            discriminator_optimizer,
            batch_size,
            epochs,
            input_size,
            input_size,
        )
        gan._generator = gan.make_generator_model()
        gan._discriminator = gan.make_discriminator_model()

        gan._param_split = len(gan._generator.get_weights())

        return gan

    @tf.function
    def train_step(self, data):
        self._logger.info("Start GAN training...")
        noise = tf.random.normal((1, 65))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self._generator(noise, training=True)

            real_output = self._discriminator(
                tf.reshape(data, [1, 65]), training=True
            )  # TODO Bug fixes
            fake_output = self._discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

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
        self._logger.info("Finished tain_step of generative model")

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def create_synthetic_data(self, n: int = 100):
        """Create n numbers of synthetic data points. These datapoints should be mixed with the locally arriving
        datastream to finetune the local intrusion detection model."""
        # TODO Seraphin: Implement generation of n synthetic data points

        generated_data = []
        for i in range(n):
            noise = tf.random.normal([1, 65])
            generated_data.append(self._generator(noise, training=True))

        generated_data_tensor = tf.concat(generated_data, axis=0)
        self._logger.warning("Generated Synthetic data...")

        # df = pd.DataFrame(generated_data_tensor)
        # correlation_matrix = df.corr()
        # fig = plt.figure(figsize=(10, 8))
        # sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f')
        # plt.title('Correlation Heatmap')
        # fig.savefig('synthetic_correlation.png', dpi=fig.dpi)
        # self._logger.info("Synthetic data written to file...")
        return generated_data_tensor

    def fit(self, dataset):
        """Train Generator and Discriminator of generateive GAN"""
        self._logger.info("Fit called...")
        for epoch in range(self._epochs):
            for data_point in dataset:
                self.train_step(data_point)

    def predict(self):
        raise NotImplementedError
