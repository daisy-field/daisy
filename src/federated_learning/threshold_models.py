"""
    A collection of extensions to the FederatedModel base class to support a wide array of threshold models, which are
    used for anomaly detection, mostly for the mapping from numerical scalar values to binary class labels.

    Author: Fabian Hofmann, Seraphin Zunzer
    Modified: 11.09.23
"""
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from src.federated_learning.federated_model import FederatedModel


class FederatedTM(FederatedModel, ABC):
    """Abstract base class for federated threshold models, all of which, to perform predictions, simply compare the
    current samples to a dynamic threshold that is updated using internal parameters, which vary between approaches/
    implementations.
    """
    _threshold: float
    _reduce_fn: Callable

    def __init__(self, threshold: float = 0, reduce_fn: Callable[[Tensor], Tensor] = lambda o: o):
        """Creates a new threshold model.

        :param threshold: Init for actual threshold value the module uses to process samples.
        :param reduce_fn: Function to reduce a batch of samples into a scalar for training. Defaults to NOP.
        """
        self._threshold = threshold
        self._reduce_fn = reduce_fn

    def set_parameters(self, parameters: list[np.ndarray]):
        """Updates the internal parameters of the threshold model, before re-computing the actual threshold value based
        on the internal parameters.

        Important Note: Must be overriden (and called afterward) when class is implemented.

        :param parameters: Parameters to update the threshold model with.
        """
        self._threshold = self.update_threshold()

    @abstractmethod
    def get_parameters(self) -> list[np.ndarray]:
        """Retrieves the internal parameters of the threshold model, actual threshold excluded.

        :return: Parameters of threshold model.
        """
        raise NotImplementedError

    def fit(self, x_data, y_data=None):
        """Trains the threshold model with the given data, which must be compatible with the tensorflow API
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit). If the input data contains more than one
        sample, it is reduced to a singular scalar using a set function, if defined (see: __init__). This is akin to
        an additional layer of window-based smoothing/averaging, also speeding up the update process.

        :param x_data: Input data.
        :param y_data: Expected output, optional since most (simple) threshold models are unsupervised.
        """
        reduced_data = self._reduce_fn(x_data)
        self._threshold = self.update_threshold(reduced_data)

    def predict(self, x_data) -> Tensor[bool]:
        """Makes a prediction on the given data and returns it, which must be compatible with the tensorflow API
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).

        :param x_data: Input data.
        :return: Predicted output tensor consisting of bools (0: normal, 1: abnormal).
        """
        return tf.math.greater(x_data, self._threshold)

    @abstractmethod
    def update_threshold(self, x_data=None) -> float:
        """Updates the internal parameters of the threshold model using a batch of new samples to compute the new
        threshold value.

        :param x_data: Batch of input data. Optional.
        :return: New threshold value.
        """
        raise NotImplementedError

class AvgTM(FederatedTM):
    """TODO
    avg based threshold model that bbuilds the threshold by aggr

    """
    _mean: float
    _var: float

    def __init__(self):
        pass

    def set_parameters(self, parameters: list[np.ndarray]):
        pass

    def get_parameters(self) -> list[np.ndarray]:
        pass

    def update_threshold(self, x_data=None) -> float:
        pass







# def update(existing_aggregate, new_value):
#     (count, mean, M2) = existing_aggregate
#     count += 1
#     delta = new_value - mean
#     mean += delta / count
#     delta2 = new_value - mean
#     M2 += delta * delta2
#     return (count, mean, M2)
