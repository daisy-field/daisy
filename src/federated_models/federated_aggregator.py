"""
    TODO

    Author: Fabian Hofmann, Seraphin Zunzer
    Modified: 14.08.23
"""

from abc import ABC, abstractmethod

import numpy as np


class Aggregator(ABC):
    """
    TODO
    """

    @abstractmethod
    def aggregate(self) -> object:
        """TODO

        :return:
        """
        raise NotImplementedError


class FedAvgAggregator(Aggregator):
    """TODO
    
    :param ModelAggregator: 
    :return: 
    """

    def aggregate(self, *models: list[np.ndarray]) -> list[np.ndarray]:
        """FIXME
        Calculate fedavg between global weights and new weights.
        When weights are malformed, skip aggregation and return old weights.

        Kahan summation algorithm vs Welford

        :param global_weights: Weights of global model
        :param client_weights: Weights of client model that should be aggregated
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
