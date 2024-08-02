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
"""
# TODO: Future Work:
#   - weighted moving average based on significance of data point
#   - in that case requires serious consideration whether a list of model parameters
#     can even work
#   - should be done with a custom weighting interface that computes the importance
#     of model to the set (0-1)

from abc import ABC, abstractmethod
from collections import deque
from keras import Model
from difflib import SequenceMatcher

import numpy as np
import json


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
        print(self._n)
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


class LCAggregator(ModelAggregator):
    _commonalities = {}

    def __init__(self, models: dict[int, list[Model]]):
        """ Creates a new layerwise aggregator.

        :param layers: layers of the initial models
        """
        self._fed_avg = FedAvgAggregator()

        for model_key in models.keys():
            # Build the information about the model layers of the current archetype
            layers = self.build_own_information(models[model_key], model_key)

            # Go through all other models and add their respective similarities
            for other_key in self._commonalities.keys():
                if other_key == model_key:
                    continue

                # Find layerwise commonalities between two model architectures
                other_layers = self._commonalities[other_key]['layers']
                sequence_len, sequence_start_own, sequence_start_other = (
                    self.find_layer_similarities(layers, other_layers))

                # Get indices of layers to share
                shareable_layers = np.arange(sequence_start_own, sequence_start_own + sequence_len, dtype=int)
                self.get_relevant_weights((model_key, other_key), shareable_layers)

                shareable_layers = np.arange(sequence_start_other, sequence_start_other + sequence_len, dtype=int)
                self.get_relevant_weights((other_key, model_key), shareable_layers)

        print(self._commonalities)

    def build_own_information(self, model, model_key) -> list[str]:
        # Construct the layer list with information for each layer
        layers = []
        weight_indices = []
        for i, layer in enumerate(model):
            layers.append(f'{layer.__class__.__name__};{layer.count_params()}')
            weight_indices.extend([i] * len(layer.get_weights()))
        weight_indices = np.asarray(weight_indices)

        # Add yourself to the knowledge base
        self._commonalities[model_key] = {
            'layers': layers,
            'weight_indices': weight_indices,
        }

        return layers

    @staticmethod
    def find_layer_similarities(own_layers: list[str], other_layers: list[str]) -> (int, int, int):
        match = SequenceMatcher(None, own_layers, other_layers).find_longest_match()
        return match.size, match.a, match.b

    def get_relevant_weights(self, model_ids: (int, int), layer_indices: np.ndarray):
        occurrences = []
        for index in layer_indices:
            occurrences.extend(np.where(index == self._commonalities[model_ids[0]]['weight_indices'])[0].tolist())
        self._commonalities[model_ids[0]][model_ids[1]] = occurrences

    def aggregate(self, models_parameters: list[(int, list[np.ndarray])]) -> list[list[np.ndarray]]:
        aggregated_models = []
        for (i, (first_id, first_weights)) in enumerate(models_parameters):
            aggr = CumAggregator()
            temp = aggr.aggregate([first_weights])
            for (j, (second_id, second_weights)) in enumerate(models_parameters):
                if i == j:
                    continue
                elif first_id == second_id:
                    temp = aggr.aggregate([second_weights])
                else:
                    # Get the relevant weights from the second model
                    relevant_layers = self._commonalities[second_id][first_id]
                    relevant_weights = [second_weights[idx] for idx in relevant_layers]
                    # Replace the relevant layers with weights from the second model
                    second_weights = temp.copy()
                    relevant_layers = self._commonalities[first_id][second_id]
                    for k, idx in enumerate(relevant_layers):
                        second_weights[idx] = relevant_weights[k]
                    temp = aggr.aggregate([second_weights])
            aggregated_models.append(temp)
        return aggregated_models
