"""
    A collection of various types of model wrappers, implementing the same interface for each federated model type, thus
    enabling their inter-compatibility for different aggregation strategies and federated system components.

    Author: Fabian Hofmann, Seraphin Zunzer
    Modified: 14.08.23
"""
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from tensorflow import keras


class FederatedModel(ABC):
    """An abstract model wrapper that offers the same methods, no matter the type of underlying model. Must always be
    implemented if a new model type is to be used in the federated system.
    """

    @abstractmethod
    def update_parameters(self, parameters: list[np.ndarray]):
        """Updates the internal parameters of the model.

        :param parameters: Parameters to update the model with.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_parameters(self) -> list[np.ndarray]:
        """Retrieves the internal parameters of the model.

        :return: Parameters of model.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, x_data, y_data):
        """Trains the model with the given data.

        :param x_data: Input data.
        :param y_data: Expected output.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_data) -> np.ndarray:
        """Makes a prediction on the given data and returns it.

        :param x_data: Input data.
        :return: Predicted output.
        """
        raise NotImplementedError


class TFFederatedModel(FederatedModel):
    """The standard federated model wrapper for tensorflow models. Can be used for both online and offline training/
    prediction, by default online, however.
    """
    _model: keras.Model
    _batch_size: int
    _epochs: int

    def __init__(self, model: keras.Model, optimizer: str | keras.optimizers, loss: str | keras.losses.Loss,
                 metrics: list[str | Callable, keras.metrics.Metric] = None,
                 batch_size: int = 32, epochs: int = 1):
        """Creates a new tensorflow federated model from a given model. Since this also compiles the given model,
        there are set of additional arguments, for more information on those see:
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile

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

    def update_parameters(self, parameters: list[np.ndarray]):
        """Updates the weights of the underlying model with new ones.

        :param parameters: Weights to update the model with.
        """
        self._model.set_weights(parameters)

    def retrieve_parameters(self) -> list[np.ndarray]:
        """Retrieves the weights of the underlying model.

        :return: Weights of the model.
        """
        return self._model.get_weights()

    def fit(self, x_data, y_data):
        """Trains the model with the given data.

        :param x_data: Input data.
        :param y_data: Expected output.
        """
        self._model.fit(x=x_data, y=y_data, batch_size=self._batch_size, epochs=self._epochs)

    def predict(self, x_data) -> np.ndarray:
        """Makes a prediction on the given data and returns it. Uses the call() tensorflow model interface for small
        numbers of data points.

        :param x_data: Input data.
        :return: Predicted output.
        """
        if len(x_data) > self._batch_size:
            return self._model.predict(x=x_data, batch_size=self._batch_size)
        return self._model(x_data, training=False).numpy()
