"""
    To allow the compatibility between different kinds of strategies for the aggregation of models, gradients, losses,
    or other parameters and topologies for the federated distributed system, any strategy has to follow the same simple
    interface. This module provides this abstract class and some sample aggregators to be used.

    Author: Fabian Hofmann, Seraphin Zunzer
    Modified: 14.08.23
"""

from abc import ABC, abstractmethod

import numpy as np


class Aggregator(ABC):
    """Abstract aggregator class. Must always be implemented if a new aggregation strategy is used in the federated
    system.
    """

    @abstractmethod
    def aggregate(self) -> object:
        """Calculates the aggregated values from the underlying object attributes and potential given parameters and
        returns them.

        :return: Aggregated values.
        """
        raise NotImplementedError


class FedAvgAggregator(Aggregator):
    """Federated Averaging (FedAvg) is the simplest way to aggregate any number of models into a "global" model version,
    using only their weights; simply computing their average and using that as a result.
    Note this could also be done using gradients or losses or any other form of parameters from other types of models,
    however this was implemented with the federated_model.TFFederatedModel in mind.
    """

    def aggregate(self, *models: list[np.ndarray]) -> list[np.ndarray]:
        """Calculates the average for the given models' parameters and returns it, using Welford's algorithm.
        Note this could be further optimized using Kahan–Babuška summation for reduction of floating point errors.

        :param models: Weights of global model
        :return: New weights for global model
        """
        avg_parameters = models[0]
        count = 0
        for model in models:
            count += 1
            for i in range(len(model)):
                delta = model[i] - avg_parameters[i]
                avg_parameters[i] += delta / count
        return avg_parameters
