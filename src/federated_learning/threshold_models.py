"""
    A collection of extensions to the FederatedModel base class to support a wide array of threshold models, which are
    used for anomaly detection, mostly for the mapping from numerical scalar values to binary class labels. For this the
    most common models currently are the statistical ones, that use either the mean (combined with std. dev.) or the
    median to compute a singular threshold value for simple classification.

    Author: Fabian Hofmann, Seraphin Zunzer
    Modified: 14.09.23
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
    current samples (which are in order to a dynamic threshold that is updated using internal parameters, which vary
    between approaches/implementations.

    Note this kind of model *absolutely requires* a time series in most cases and therefore data passed to it, must be
    in order!

    Note that for the initial value, the threshold is always zero (any point is considered an anomaly during detection).
    """
    _threshold: float
    _reduce_fn: Callable

    def __init__(self, threshold: float = 0, reduce_fn: Callable[[Tensor], Tensor] = lambda o: o):
        """Creates a new threshold model.

        :param threshold: Init for actual threshold value.
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
    """Base class for average-based threshold models, all of which computing their internal threshold values by first
    computing the mean and standard deviation for the data stream and then adding them together for the (absolute)
    threshold of the model. This method is very similar to the one employed by Florian et al. for the original version
    of IFTM as well (https://ieeexplore.ieee.org/document/8456348), and was therefore implemented here, however this
    approach is not alone for error-based anomaly detection approaches, since it follows simple statistical assumptions
    for normal distributions (i.e., a sample is considered anomalous if it is further than x-times the std. dev. from
    the mean of the total population), being very similar to average absolute deviation methods (AAD).

    Any implementation of this class must provide a way to update the mean using new incoming samples, anything else is
    already taken care of by this base class.

    Note that many of the implementations are very similar to the ModelAggregator implementations, as both treat the
    aggregated values as a timeseries
    """
    _mean: float
    _var: float
    _var_weight: float

    def __init__(self, mean: float = None, var: float = None, var_weight: float = 1.0,
                 reduce_fn: Callable[[Tensor], Tensor] = lambda o: o):
        """Creates a new average-based threshold model.

        :param mean: Init mean value.
        :param var: Init variance value.
        :param var_weight: Weight of the variance compared to the mean value for the threshold value.
        :param reduce_fn: Function to reduce a batch of samples into a scalar for training. Defaults to NOP.
        """
        self._mean = mean
        self._var = var
        self._var_weight = var_weight
        super().__init__(threshold=0, reduce_fn=reduce_fn)

    def set_parameters(self, parameters: list[np.ndarray]):
        """Updates the mean and variance of the threshold model, then updates the model (adjusting threshold).

        Note the nested method arguments --- for the base class the first position of the list is allocated for the
        statistical values, while extending classes use the list's positions that come after.

        :param parameters: Mean and variance to update threshold model with.
        """
        self._mean = cast(float, parameters[0][0])
        self._var = cast(float, parameters[0][1])
        super().set_parameters(parameters)

    def get_parameters(self) -> list[np.ndarray]:
        """Retrieves the mean and variance of the threshold model, actual threshold excluded.

        :return: Mean and variance of threshold model.
        """
        return [np.array([self._mean, self._var], dtype=np.float32)]

    def update_threshold(self, x_data=None):
        """Updates the mean and variance of the threshold model using a batch of new samples to compute the new
        threshold value. If none provided, simply re-compute the threshold based on the current values.

        :param x_data: Batch of input data. Optional.
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
        """Updates the internal mean value with new sample.

        :param new_sample: New sample of time series.
        """
        raise NotImplementedError


class CumAvgTM(AvgTM):
    """Cumulative Averaging is the base version of online averaging, which is equal to the offline average after seeing
    every instance of the population.

    Consequently, this aggregator is NOT stable for infinite learning aggregation steps (n).
    """
    _n: int

    def __init__(self, mean: float = None, var: float = None, var_weight: float = 1.0,
                 reduce_fn: Callable[[Tensor], Tensor] = lambda o: o):
        """Creates a new cumulative averaging threshold model.

        :param mean: Init mean value.
        :param var: Init variance value.
        :param var_weight: Weight of the variance compared to the mean value for the threshold value.
        :param reduce_fn: Function to reduce a batch of samples into a scalar for training. Defaults to NOP.
        """
        self._n = 0
        super().__init__(mean=mean, var=var, var_weight=var_weight, reduce_fn=reduce_fn)

    def set_parameters(self, parameters: list[np.ndarray]):
        """Updates the parameters of the cumulative averaging threshold model, which includes the size of the current
        population. The rest of the updating is done as described in the superclass' method.

        :param parameters: Parameters to update threshold model with, with the population number in second place.
        """
        self._n = cast(int, parameters[1][0])
        super().set_parameters(parameters)

    def get_parameters(self) -> list[np.ndarray]:
        """Retrieves the parameters of the cumulative averaging threshold model, actual threshold excluded.

        :return: Mean and variance (1st) and size of population (2nd) of threshold model.
        """
        return (super().get_parameters()
                + [np.array([self._n], dtype=np.float32)])

    def update_mean(self, new_sample: float):
        """Updates the cumulative mean value with new sample.

        :param new_sample: New sample of time series.
        """
        self._n += 1
        if self._n == 1:
            self._mean = new_sample
        else:
            delta = new_sample - self._mean
            self._mean += delta / self._n


class SMAvgTM(AvgTM):
    """Simple Moving Averaging, also called sliding window averaging, is the simplest version of online moving
    averaging, as it uses only the past k elements of the population to compute the average. Note that this really needs
    a proper datastream/timeseries, as the order of samples influences the average at every step.
    """
    _window: deque
    _window_size: int

    def __init__(self, window_size: int = 5, mean: float = None, var: float = None, var_weight: float = 1.0,
                 reduce_fn: Callable[[Tensor], Tensor] = lambda o: o):
        """Creates a new simple moving averaging threshold model.

        :param window_size: Size of sliding window.
        :param mean: Init mean value.
        :param var: Init variance value.
        :param var_weight: Weight of the variance compared to the mean value for the threshold value.
        :param reduce_fn: Function to reduce a batch of samples into a scalar for training. Defaults to NOP.
        """
        self._window = deque()
        self._window_size = window_size
        super().__init__(mean=mean, var=var, var_weight=var_weight, reduce_fn=reduce_fn)

    def set_parameters(self, parameters: list[np.ndarray]):
        """Updates the parameters of the cumulative averaging threshold model, which includes the current content of the
        sliding window. The rest of the updating is done as described in the superclass' method.

        Note: Should only be used with threshold models that support the same window size, otherwise random things might
        happen.

        :param parameters: Parameters to update threshold model with sliding window's samples in second place.
        """
        self._window = deque()
        new_window = parameters[1]
        for sample in new_window:
            self._window.append(sample)
        super().set_parameters(parameters)

    def get_parameters(self) -> list[np.ndarray]:
        """Retrieves the parameters of the cumulative averaging threshold model, actual threshold excluded.

        :return: Mean and variance (1st) and sliding window (2nd) of threshold model.
        """
        return (super().get_parameters()
                + [np.array(self._window, dtype=np.float32)])

    def update_mean(self, new_sample: float):
        """Updates the simple moving average value with new sample, removing the last sample from the sliding window and
        storing the new one in it, adjusting the mean accordingly.

        :param new_sample: New sample of time series.
        """
        if len(self._window) == 0:
            self._mean = new_sample
            return

        rm_sample = 0
        if len(self._window) >= self._window_size:
            rm_sample = self._window.popleft()
        self._window.append(new_sample)

        delta = new_sample - rm_sample
        self._mean += delta / len(self._window)


class EMAvgTM(AvgTM):
    """Exponential moving averaging takes note of the entire model stream to compute the average, but weights them
    exponentially less the greater their distance to the present is; this process is also called exponential smoothing.
    """
    _alpha = float

    def __init__(self, alpha: float = 0.05, mean: float = None, var: float = None, var_weight: float = None,
                 reduce_fn: Callable[[Tensor], Tensor] = lambda o: o):
        """Creates a new exponential moving averaging threshold model.

        :param alpha: Smoothing/weight factor for new incoming values.
        :param mean: Init mean value.
        :param var: Init variance value.
        :param var_weight: Weight of the variance compared to the mean value for the threshold value.
        :param reduce_fn: Function to reduce a batch of samples into a scalar for training. Defaults to NOP.
        """
        self._alpha = alpha
        super().__init__(mean=mean, var=var, var_weight=var_weight, reduce_fn=reduce_fn)

    def update_mean(self, new_sample: float):
        """Updates the exponential moving average value with new sample, weighting past and present as defined.

        :param new_sample: New sample of time series.
        """
        if self._mean is None:
            self._mean = new_sample
        self._mean = self._alpha * new_sample + (1 - self._alpha) * self._mean


class MadTM(FederatedTM):
    """Median absolute deviation (MAD)-based threshold models, similar to AAD-based models, again assume a symmetric
    distribution of the time serie's samples, of which a certain percentage are considered anomalous. However, unlike
    average-based approaches, the median can only computed using a subset of the population when computed online (it
    cannot be computed online in fact, as this is a property of the median).
    """
    _window: deque
    _window_size: int

    def __init__(self, window_size: int = 5, reduce_fn: Callable[[Tensor], Tensor] = lambda o: o):
        """Creates a new MAD threshold model.

        :param window_size: Size of sliding window.
        :param reduce_fn: Function to reduce a batch of samples into a scalar for training. Defaults to NOP.
        """
        self._window = deque()
        self._window_size = window_size
        super().__init__(threshold=0, reduce_fn=reduce_fn)

    def set_parameters(self, parameters: list[np.ndarray]):
        """Updates the parameters of the MAD threshold model, which includes the current content of the sliding window,
        before updating the actual model (adjusting threshold).

        Note: Should only be used with threshold models that support the same window size, otherwise random things might
        happen.

        :param parameters: New sample window of time stream.
        """
        self._window = deque()
        new_window = parameters[0]
        for sample in new_window:
            self._window.append(sample)
        super().set_parameters(parameters)

    def get_parameters(self) -> list[np.ndarray]:
        """Retrieves the entire sample window of the MAD threshold model, actual threshold excluded.

        :return: Current sample window of time stream.
        """
        return [np.array(self._window, dtype=np.float32)]

    def update_threshold(self, x_data=None):
        """Updates the MAD threshold model using a batch of new samples to compute the new threshold value by adjusting
        the sliding window over the time series accordingly, re-computing the medians and the following MAD score. If no
        new data provided, simply re-compute the threshold based on the current samples inside the window.

        :param x_data: Batch of input data. Optional.
        """
        if x_data is not None:
            for sample in x_data:
                if len(self._window) == self._window_size:
                    self._window.popleft()
                self._window.append(sample)

        if len(self._window) > 0:
            samples = np.array(self._window)
            m = np.median(samples)
            ad = np.abs(samples - m)
            mad = np.median(ad)
            self._threshold = 0.6745 * ad / mad
