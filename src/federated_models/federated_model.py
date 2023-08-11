"""
    Interface for federated models. TODO

    Author: Fabian Hofmann, Seraphin Zunzer
    Modified: 11.08.23
"""
from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from tensorflow import keras


class FederatedModel(ABC):
    @abstractmethod
    def aggregate(self):
        """
        Update Model with parameters, somehow
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, x_data, y_data):
        """
        fit model with training data, somehow
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_data) -> np.ndarray:
        """
        make a prediction, somehow
        :return:
        """
        raise NotImplementedError


class TFFederatedModel(FederatedModel):
    _model: keras.Model
    _batch_size: int

    def __init__(self):
        self._model.compile()  # TODO
        pass

    def aggregate(self, *models: Self):
        raise NotImplementedError  # TODO

    def fit(self, x_data, y_data):
        self._model.fit(x_data, y_data)  # TODO

    def predict(self, x_data) -> np.ndarray:
        if len(x_data) > self._batch_size:
            return self._model.predict(x_data)  # TODO
        return self._model(x_data, training=False).numpy()



def aggregate(self, global_weights, client_weights):
    """FIXME
    Calculate fedavg between global weights and new weights.
    When weights are malformed, skip aggregation and return old weights.

    :param global_weights: Weights of global model
    :param client_weights: Weights of client model that should be aggregated
    :return: New weights for global model
    """
    old_weights = global_weights
    try:
        for i in range(0, len(global_weights)):
            for j in range(0, len(global_weights[i])):
                global_weights[i][j] = (global_weights[i][j] + client_weights[i][j]) / 2

        return global_weights
    except:
        logging.warning("Malformed weights received!")
        return old_weights