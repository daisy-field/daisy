"""
    A collection of extensions to the FederatedModel base class to support a wide array of threshold models, which are
    used for anomaly detection, mostly for the mapping from numerical scalar values to binary class labels.

    Author: Fabian Hofmann, Seraphin Zunzer
    Modified: 11.09.23
"""
from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, cast

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
        self.update_threshold()

    def set_parameters(self, parameters: list[np.ndarray]):
        """Updates the internal parameters of the threshold model, before re-computing the actual threshold value based
        on the internal parameters.

        Important Note: Must be overriden (and called afterward) when class is implemented.

        :param parameters: Parameters to update the threshold model with.
        """
        self.update_threshold()

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
        self.update_threshold(reduced_data)

    def predict(self, x_data) -> Tensor[bool]:
        """Makes a prediction on the given data and returns it, which must be compatible with the tensorflow API
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).

        :param x_data: Input data.
        :return: Predicted output tensor consisting of bools (0: normal, 1: abnormal).
        """
        return tf.math.greater(x_data, self._threshold)

    @abstractmethod
    def update_threshold(self, x_data=None):
        """Updates the internal parameters of the threshold model using a batch of new samples to compute the new
        threshold value. If none provided, simply re-compute the threshold.

        :param x_data: Batch of input data. Optional.
        """
        raise NotImplementedError


class AvgTM(FederatedTM, ABC):
    """TODO
    avg based threshold model that builds the threshold by aggr

    basically federated aggregator

    """
    _mean: float
    _var: float
    _var_weight: float

    def __init__(self, mean: float = None, var: float = None, var_weight: float = None,
                 reduce_fn: Callable[[Tensor], Tensor] = lambda o: o):
        """TODO

        must be extended.

        :param mean:
        :param var:
        :param var_weight:
        :param reduce_fn:
        """
        self._mean = mean
        self._var = var
        self._var_weight = var_weight
        super().__init__(threshold=0, reduce_fn=reduce_fn)

    def set_parameters(self, parameters: list[np.ndarray]):
        """TODO

        could be extended by implementation for more parms

        :param parameters:
        """
        self._mean = cast(float, parameters[0][0])
        self._var = cast(float, parameters[0][1])
        super().set_parameters(parameters)

    def get_parameters(self) -> list[np.ndarray]:
        """TODO

        could be extended by implementation for more parms

        :return:
        """
        return [np.array([self._mean, self._var], dtype=np.float32)]

    def update_threshold(self, x_data=None):
        """TODO

        no need to extend

        :param x_data:
        """
        if x_data is not None:
            for sample in x_data:
                d_1 = sample - self._mean
                self._mean = self.update_mean(sample)
                d_2 = sample - self._mean
                self._var += d_1 * d_2
        self._threshold = self._mean + self._var * self._var_weight if self._mean is not None else 0

    @abstractmethod
    def update_mean(self, new_sample: float):
        """TODO

        :param new_sample:
        """
        raise NotImplementedError


class CumAvgTM(AvgTM):
    """TODO

    """
    _n: int

    def __init__(self, mean: float = None, var: float = None, var_weight: float = None,
                 reduce_fn: Callable[[Tensor], Tensor] = lambda o: o):
        """TODO

        :param mean:
        :param var:
        :param var_weight:
        :param reduce_fn:
        """
        self._n = 0
        super().__init__(mean=mean, var=var, var_weight=var_weight, reduce_fn=reduce_fn)

    def update_mean(self, new_sample: float):
        """TODO

        :param new_sample:
        """
        self._n += 1
        if self._n == 1:
            self._mean = new_sample
        else:
            delta = new_sample - self._mean
            self._mean += delta / self._n


class SMAvgTM(AvgTM):
    """TODO

    """
    _window: deque
    _window_size: int

    def __init__(self, window_size: int = 5, mean: float = None, var: float = None, var_weight: float = None,
                 reduce_fn: Callable[[Tensor], Tensor] = lambda o: o):
        """TODO

        :param mean:
        :param var:
        :param var_weight:
        :param reduce_fn:
        """
        self._window = deque()
        self._window_size = window_size
        super().__init__(mean=mean, var=var, var_weight=var_weight, reduce_fn=reduce_fn)

    def update_mean(self, new_sample: float):
        """TODO

        :param new_sample:
        """
        if len(self._window) == 0:
            self._mean = new_sample
            return

        rm_sample = 0
        if len(self._window) == self._window_size:
            rm_sample = self._window.popleft()
        self._window.append(new_sample)

        delta = new_sample - rm_sample
        self._mean += delta / len(self._window)


class EMAvgTM(AvgTM):
    """TODO

    """
    _alpha = float

    def __init__(self, alpha: float = 0.05, mean: float = None, var: float = None, var_weight: float = None,
                 reduce_fn: Callable[[Tensor], Tensor] = lambda o: o):
        """TODO

        :param alpha: Smoothing/weight factor for new incoming values.
        :param mean:
        :param var:
        :param var_weight:
        :param reduce_fn:
        """
        self._alpha = alpha
        super().__init__(mean=mean, var=var, var_weight=var_weight, reduce_fn=reduce_fn)

    def update_mean(self, new_sample: float):
        """TODO

        :param new_sample:
        """
        if self._mean is None:
            self._mean = new_sample
        self._mean = self._alpha * new_sample + (1 - self._alpha) * self._mean


class MadTM(FederatedTM):
    """

    """
    _median: float
    _window: deque
    _window_size: int

    def __init__(self, window_size: int = 5, reduce_fn: Callable[[Tensor], Tensor] = lambda o: o):
        """

        :param window_size:
        :param reduce_fn:
        """
        self._window = deque()
        self._window_size = window_size
        super().__init__(threshold=0, reduce_fn=reduce_fn)

    def set_parameters(self, parameters: list[np.ndarray]):
        """TODO

        could be extended by implementation for more parms

        :param parameters:
        """
        self._median = cast(float, parameters[0][0])
        super().set_parameters(parameters)

    def get_parameters(self) -> list[np.ndarray]:
        """TODO

        could be extended by implementation for more parms

        :return:
        """
        return [np.array([self._median], dtype=np.float32)]

    def update_threshold(self, x_data=None):
        """TODO

        :param x_data:
        :return:
        """
        if x_data is not None:
            for sample in x_data:
                if len(self._window) == 0:
                    self._median = sample
                    continue
                if len(self._window) == self._window_size:
                    self._window.popleft()
                self._window.append(sample)

        if len(self._window) > 0:
            samples = np.array(self._window)
            m = np.median(samples)
            ad = np.abs(samples - m)
            mad = np.median(ad)
            self._threshold = 0.6745 * ad / mad
