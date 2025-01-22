# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Implementation of different sized models used for personalized FL

Author: Seraphin Zunzer
Modified: 24.07.24
"""

from typing import Callable, Self

import numpy as np
from tensorflow import Tensor
from tensorflow import keras

from daisy.federated_learning import FederatedModel


class TFFederatedModel_small(FederatedModel):
    """Small federated model"""

    _model: keras.Model
    _batch_size: int
    _epochs: int

    def __init__(
        self,
        model: keras.Model,
        optimizer: str | keras.optimizers.Optimizer,
        loss: str | keras.losses.Loss,
        metrics: list[str | Callable | keras.metrics.Metric] = None,
        batch_size: int = 32,
        epochs: int = 1,
    ):
        """Creates a new tensorflow federated model from a given model. This also
        compiles the given model, requiring a set of additional arguments
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile).

        :param model: Underlying model to be wrapped around.
        :param optimizer: Optimizer to use during training.
        :param loss: Loss function to use during training.
        :param metrics: Evaluation metrics to be displayed during training and testing.
        :param batch_size: Batch size during training and prediction.
        :param epochs: Number of epochs (rounds) during training.
        """
        self._model = model
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self._batch_size = batch_size
        self._epochs = epochs

    def set_parameters(self, parameters: list[np.ndarray]):
        """Updates the weights of the underlying model with new ones.

        :param parameters: Weights to update the model with.
        """
        self._model.set_weights(parameters)

    def get_parameters(self) -> list[np.ndarray]:
        """Retrieves the weights of the underlying model.

        :return: Weights of the model.
        """
        return self._model.get_weights()

    def fit(self, x_data, y_data):
        """Trains the model with the given data by calling the wrapped model,
        which must be compatible with the tensorflow API
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).

        :param x_data: Input data.
        :param y_data: Expected output.
        """
        self._model.fit(
            x=x_data, y=y_data, batch_size=self._batch_size, epochs=self._epochs
        )

    def predict(self, x_data) -> Tensor:
        """Makes a prediction on the given data and returns it by calling the wrapped
        model. Uses the call() tensorflow model interface for small numbers of data
        points, which must be compatible with the tensorflow API (see:
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict).

        :param x_data: Input data.
        :return: Predicted output tensor.
        """
        if len(x_data) > self._batch_size:
            return self._model.predict(x=x_data, batch_size=self._batch_size)
        return self._model(x_data, training=False).numpy()

    @classmethod
    def get_fae(
        cls,
        input_size: int,
        optimizer: str | keras.optimizers.Optimizer = "Adam",
        loss: str | keras.losses.Loss = "mse",
        metrics: list[str | Callable | keras.metrics.Metric] = None,
        batch_size: int = 32,
        epochs: int = 1,
    ) -> Self:
        """Factory class method to create a simple federated autoencoder model of a
        fixed depth but with variable input size.

        Should only serve as a quick and basic setup for a model.

        :param input_size: Dimensionality of input/output of autoencoder.
        :param optimizer: Optimizer to use during training.
        :param loss: Loss function to use during training.
        :param metrics: Evaluation metrics to be displayed during training and testing.
        :param batch_size: Batch size during training and prediction.
        :param epochs: Number of epochs (rounds) during training.
        :return: Initialized federated autoencoder model.
        """
        enc_inputs = keras.layers.Input(shape=(input_size,))
        x = keras.layers.Dense(input_size)(enc_inputs)
        enc_outputs = keras.layers.Dense(18)(x)
        encoder = keras.Model(inputs=enc_inputs, outputs=enc_outputs)

        dec_inputs = keras.layers.Input(shape=(18,))
        y = keras.layers.Dense(input_size)(dec_inputs)
        dec_outputs = keras.layers.Activation("sigmoid")(y)
        decoder = keras.Model(inputs=dec_inputs, outputs=dec_outputs)

        fae_inputs = keras.Input(shape=(input_size,))
        encoded = encoder(fae_inputs)
        fae_outputs = decoder(encoded)
        fae = keras.models.Model(inputs=fae_inputs, outputs=fae_outputs)
        return TFFederatedModel_small(fae, optimizer, loss, metrics, batch_size, epochs)


class TFFederatedModel_medium(FederatedModel):
    """Medium sized federated model"""

    _model: keras.Model
    _batch_size: int
    _epochs: int

    def __init__(
        self,
        model: keras.Model,
        optimizer: str | keras.optimizers.Optimizer,
        loss: str | keras.losses.Loss,
        metrics: list[str | Callable | keras.metrics.Metric] = None,
        batch_size: int = 32,
        epochs: int = 1,
    ):
        """Creates a new tensorflow federated model from a given model. This also
        compiles the given model, requiring a set of additional arguments
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile).

        :param model: Underlying model to be wrapped around.
        :param optimizer: Optimizer to use during training.
        :param loss: Loss function to use during training.
        :param metrics: Evaluation metrics to be displayed during training and testing.
        :param batch_size: Batch size during training and prediction.
        :param epochs: Number of epochs (rounds) during training.
        """
        self._model = model
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self._batch_size = batch_size
        self._epochs = epochs

    def set_parameters(self, parameters: list[np.ndarray]):
        """Updates the weights of the underlying model with new ones.

        :param parameters: Weights to update the model with.
        """
        self._model.set_weights(parameters)

    def get_parameters(self) -> list[np.ndarray]:
        """Retrieves the weights of the underlying model.

        :return: Weights of the model.
        """
        return self._model.get_weights()

    def fit(self, x_data, y_data):
        """Trains the model with the given data by calling the wrapped model,
        which must be compatible with the tensorflow API
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).

        :param x_data: Input data.
        :param y_data: Expected output.
        """
        self._model.fit(
            x=x_data, y=y_data, batch_size=self._batch_size, epochs=self._epochs
        )

    def predict(self, x_data) -> Tensor:
        """Makes a prediction on the given data and returns it by calling the wrapped
        model. Uses the call() tensorflow model interface for small numbers of data
        points, which must be compatible with the tensorflow API (see:
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict).

        :param x_data: Input data.
        :return: Predicted output tensor.
        """
        if len(x_data) > self._batch_size:
            return self._model.predict(x=x_data, batch_size=self._batch_size)
        return self._model(x_data, training=False).numpy()

    @classmethod
    def get_fae(
        cls,
        input_size: int,
        optimizer: str | keras.optimizers.Optimizer = "Adam",
        loss: str | keras.losses.Loss = "mse",
        metrics: list[str | Callable | keras.metrics.Metric] = None,
        batch_size: int = 32,
        epochs: int = 1,
    ) -> Self:
        """Factory class method to create a simple federated autoencoder model of a
        fixed depth but with variable input size.

        Should only serve as a quick and basic setup for a model.

        :param input_size: Dimensionality of input/output of autoencoder.
        :param optimizer: Optimizer to use during training.
        :param loss: Loss function to use during training.
        :param metrics: Evaluation metrics to be displayed during training and testing.
        :param batch_size: Batch size during training and prediction.
        :param epochs: Number of epochs (rounds) during training.
        :return: Initialized federated autoencoder model.
        """
        enc_inputs = keras.layers.Input(shape=(input_size,))
        x = keras.layers.Dense(input_size)(enc_inputs)
        x = keras.layers.Dense(35)(x)
        enc_outputs = keras.layers.Dense(18)(x)
        encoder = keras.Model(inputs=enc_inputs, outputs=enc_outputs)

        dec_inputs = keras.layers.Input(shape=(18,))
        y = keras.layers.Dense(35)(dec_inputs)
        y = keras.layers.Dense(input_size)(y)
        dec_outputs = keras.layers.Activation("sigmoid")(y)
        decoder = keras.Model(inputs=dec_inputs, outputs=dec_outputs)

        fae_inputs = keras.Input(shape=(input_size,))
        encoded = encoder(fae_inputs)
        fae_outputs = decoder(encoded)
        fae = keras.models.Model(inputs=fae_inputs, outputs=fae_outputs)
        return TFFederatedModel_small(fae, optimizer, loss, metrics, batch_size, epochs)


class TFFederatedModel_large(FederatedModel):
    """Large federated model"""

    _model: keras.Model
    _batch_size: int
    _epochs: int

    def __init__(
        self,
        model: keras.Model,
        optimizer: str | keras.optimizers.Optimizer,
        loss: str | keras.losses.Loss,
        metrics: list[str | Callable | keras.metrics.Metric] = None,
        batch_size: int = 32,
        epochs: int = 1,
    ):
        """Creates a new tensorflow federated model from a given model. This also
        compiles the given model, requiring a set of additional arguments
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile).

        :param model: Underlying model to be wrapped around.
        :param optimizer: Optimizer to use during training.
        :param loss: Loss function to use during training.
        :param metrics: Evaluation metrics to be displayed during training and testing.
        :param batch_size: Batch size during training and prediction.
        :param epochs: Number of epochs (rounds) during training.
        """
        self._model = model
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self._batch_size = batch_size
        self._epochs = epochs

    def set_parameters(self, parameters: list[np.ndarray]):
        """Updates the weights of the underlying model with new ones.

        :param parameters: Weights to update the model with.
        """
        self._model.set_weights(parameters)

    def get_parameters(self) -> list[np.ndarray]:
        """Retrieves the weights of the underlying model.

        :return: Weights of the model.
        """
        return self._model.get_weights()

    def fit(self, x_data, y_data):
        """Trains the model with the given data by calling the wrapped model,
        which must be compatible with the tensorflow API
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).

        :param x_data: Input data.
        :param y_data: Expected output.
        """
        self._model.fit(
            x=x_data, y=y_data, batch_size=self._batch_size, epochs=self._epochs
        )

    def predict(self, x_data) -> Tensor:
        """Makes a prediction on the given data and returns it by calling the wrapped
        model. Uses the call() tensorflow model interface for small numbers of data
        points, which must be compatible with the tensorflow API (see:
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict).

        :param x_data: Input data.
        :return: Predicted output tensor.
        """
        if len(x_data) > self._batch_size:
            return self._model.predict(x=x_data, batch_size=self._batch_size)
        return self._model(x_data, training=False).numpy()

    @classmethod
    def get_fae(
        cls,
        input_size: int,
        optimizer: str | keras.optimizers.Optimizer = "Adam",
        loss: str | keras.losses.Loss = "mse",
        metrics: list[str | Callable | keras.metrics.Metric] = None,
        batch_size: int = 32,
        epochs: int = 1,
    ) -> Self:
        """Factory class method to create a simple federated autoencoder model of a
        fixed depth but with variable input size.

        Should only serve as a quick and basic setup for a model.

        :param input_size: Dimensionality of input/output of autoencoder.
        :param optimizer: Optimizer to use during training.
        :param loss: Loss function to use during training.
        :param metrics: Evaluation metrics to be displayed during training and testing.
        :param batch_size: Batch size during training and prediction.
        :param epochs: Number of epochs (rounds) during training.
        :return: Initialized federated autoencoder model.
        """
        enc_inputs = keras.layers.Input(shape=(input_size,))
        x = keras.layers.Dense(input_size)(enc_inputs)
        x = keras.layers.Dense(60)(x)
        x = keras.layers.Dense(40)(x)
        x = keras.layers.Dense(35)(x)
        enc_outputs = keras.layers.Dense(18)(x)
        encoder = keras.Model(inputs=enc_inputs, outputs=enc_outputs)

        dec_inputs = keras.layers.Input(shape=(18,))
        y = keras.layers.Dense(35)(dec_inputs)
        y = keras.layers.Dense(40)(y)
        y = keras.layers.Dense(60)(y)
        y = keras.layers.Dense(input_size)(y)
        dec_outputs = keras.layers.Activation("sigmoid")(y)
        decoder = keras.Model(inputs=dec_inputs, outputs=dec_outputs)

        fae_inputs = keras.Input(shape=(input_size,))
        encoded = encoder(fae_inputs)
        fae_outputs = decoder(encoded)
        fae = keras.models.Model(inputs=fae_inputs, outputs=fae_outputs)
        return TFFederatedModel_large(fae, optimizer, loss, metrics, batch_size, epochs)
