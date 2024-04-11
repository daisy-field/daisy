# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""To allow the compatibility between different kinds of strategies for the
aggregation of models, gradients, losses, or other parameters/model types and
topologies for the federated distributed system, any strategy has to follow the same
simple interface. This module provides this abstract class and some sample
aggregators to be used.

Author: Fabian Hofmann
Modified: 04.04.24

TODO Future Work: weighted moving average based on significance of data point
TODO - in that case requires serious consideration whether a list of model parameters
can even work
TODO - should be done with a custom weighting interface that computes the importance
of model to the set (0-1)
"""

from abc import ABC, abstractmethod
from collections import deque

import numpy as np


class Aggregator(ABC):
    """Abstract aggregator class. Must always be implemented if a new aggregation
    strategy is used in the federated system.

    By implementing the aggregate function, allows simple aggregation of values using
    the object() call directly.
    """

    @abstractmethod
    def aggregate(self, *args, **kwargs) -> object:
        """Calculates the aggregated values from the underlying object attributes and
        potential given parameters and returns them.

        :param args: Arguments. Depends on the type of aggregator to be implemented.
        :param kwargs: Keyword arguments. Depends on the type of aggregator to be
        implemented.
        :return: Aggregated values.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> object:
        return self.aggregate(*args, **kwargs)


class ModelAggregator(Aggregator):
    """Abstract model aggregator class, that uses the parameters of the given models
    and some internal states (of those parameters) to compute the aggregated state of
    the model. Must always be implemented if a new model aggregation strategy is used
    in the federated system.
    """

    @abstractmethod
    def aggregate(self, models_parameters: list[list[np.ndarray]]) -> list[np.ndarray]:
        """Calculates the aggregated values from the underlying object attributes and
        given model parameters and returns them.

        :param models_parameters: A list of models' parameters the same type.
        :return: Aggregated model parameters.
        """
        raise NotImplementedError


class FedAvgAggregator(ModelAggregator):
    """Federated Averaging (FedAvg) is the simplest way to aggregate any number of
    models into a "global" model version, using only their weights (no internal state
    is kept); simply computing their average and using that as a result.

    Note this could also be done using gradients or losses or any other form of
    parameters from other types of models, however this was implemented with the
    federated_model.TFFederatedModel and its parameters in mind.
    """

    def aggregate(self, models_parameters: list[list[np.ndarray]]) -> list[np.ndarray]:
        """Calculates the average for the given models' parameters and returns it,
        using Welford's algorithm. Note this could be further optimized using
        Kahan–Babuška summation for reduction of floating point errors.

        :param models_parameters: A list of models' parameters the same type.
        :return: Aggregated model parameters.
        """
        avg_parameters = models_parameters[0]
        n = 0
        for model in models_parameters:
            n += 1
            for i in range(len(model)):
                delta = model[i] - avg_parameters[i]
                avg_parameters[i] += delta / n
        return avg_parameters


class CumAggregator(ModelAggregator):
    """Cumulative Averaging (FedCum) is the base version of online averaging for
    models into a "global" version of the model. The cumulative average is computed
    by first computing the FedAvg over the batch of models provided to the aggregator
    in a single step, before using that average to update the average (i.e.,
    all models in a batch are treated equally).

    This aggregator is NOT stable for infinite learning aggregation steps (n).

    This kind of aggregation is also probably not compatible with the aggregation of
    gradients or losses, as these averages cannot be computed over various epochs
    overall.
    """

    _cum_avg: list[np.ndarray]
    _n: int

    _fed_avg: FedAvgAggregator

    def __init__(self):
        """Creates a new cumulative aggregator."""
        self._n = 0

        self._fed_avg = FedAvgAggregator()

    def aggregate(self, models_parameters: list[list[np.ndarray]]) -> list[np.ndarray]:
        """Calculates the cumulative average for the given models' parameters and the
        previous cumulative average and returns it.

        :param models_parameters: A list of models' parameters the same type.
        :return: Aggregated model parameters.
        """
        models_avg = self._fed_avg.aggregate(models_parameters)

        self._n += 1
        if self._n == 1:
            self._cum_avg = models_avg
        else:
            for i in range(len(models_avg)):
                delta = models_avg[i] - self._cum_avg[i]
                self._cum_avg[i] += delta / self._n
        return self._cum_avg


class SMAggregator(ModelAggregator):
    """Simple Moving Averaging, also called sliding window averaging (FedSMA) is the
    simplest version of online moving averaging for models, as it uses only the past
    k elements of the stream to compute the average. As with other online model
    averaging aggregators, the regular FedAvg is first computed over the batch of
    modes provided in a single step, before using that average to update the moving
    average (i.e., all models in a batch are treated equally).

    This kind of aggregation is also probably not compatible with the aggregation of
    gradients or losses, as these averages cannot be computed over various epochs
    overall.
    """

    _sm_avg: list[np.ndarray]
    _window: deque
    _window_size: int

    _fed_avg: FedAvgAggregator

    def __init__(self, window_size: int = 5):
        """Creates a new simple moving aggregator.

        :param window_size: Size of sliding window.
        """
        self._window = deque()
        self._window_size = window_size

        self._fed_avg = FedAvgAggregator()

    def aggregate(self, models_parameters: list[list[np.ndarray]]) -> list[np.ndarray]:
        """Calculates the simple moving average for the given models' parameters by
        averaging them first and then adding them to the window. Afterward, the delta
        between the new added values and the first (to be removed) value from the
        window is computed to aggregate the new moving average.

        :param models_parameters: A list of models' parameters the same type.
        :return: Aggregated model parameters.
        """
        models_avg = self._fed_avg.aggregate(models_parameters)
        if len(self._window) == 0:
            self._sm_avg = models_avg
            return self._sm_avg

        rm_models_avg = 0
        if len(self._window) == self._window_size:
            rm_models_avg = self._window.popleft()
        self._window.append(models_avg)

        for i in range(len(models_avg)):
            delta = models_avg[i] - rm_models_avg[i]
            self._sm_avg[i] += delta / len(self._window)
        return self._sm_avg


class EMAggregator(ModelAggregator):
    """Exponential moving averaging (FedEMA) unlike FedSMA takes note of the entire
    model stream to compute the average, but over multiple time steps starts to
    forget past elements; this process is also called exponential smoothing,
    as the past is exponentially less relevant to the current time step the further
    one goes back. As with other online model averaging aggregators, the regular
    FedAvg is first computed over the batch of modes provided in a single step,
    before using that average to update the moving average (i.e., all models in a
    batch are treated equally).

    This kind of aggregation is also probably not compatible with the aggregation of
    gradients or losses, as these averages cannot be computed over various epochs
    overall.
    """

    _em_avg = list[np.ndarray]
    _alpha = float

    _fed_avg: FedAvgAggregator

    def __init__(self, alpha: float = 0.05):
        """Creates a new exponential moving aggregator.

        :param alpha: Smoothing/weight factor for new incoming values.
        """
        self._em_avg = None
        self._alpha = alpha

        self._fed_avg = FedAvgAggregator()

    def aggregate(self, models_parameters: list[list[np.ndarray]]) -> list[np.ndarray]:
        """Calculates the exponential moving average for the given models' parameters
        and returns it

        :param models_parameters: A list of models' parameters the same type.
        :return: Aggregated model parameters.
        """
        models_avg = self._fed_avg.aggregate(models_parameters)

        if self._em_avg is None:
            self._em_avg = models_avg

        for i in range(len(models_avg)):
            self._em_avg[i] = (
                self._alpha * models_avg[i] + (1 - self._alpha) * self._em_avg[i]
            )
        return self._em_avg
