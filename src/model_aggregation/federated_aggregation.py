"""
    Interface for federated aggregation methods.

    Author: Seraphin Zunzer
    Modified: 09.08.23
"""
from abc import ABC, abstractmethod


class FederatedAggregation(ABC):

    @abstractmethod
    def aggregate(self, *args, **kwargs):
        """
        Function to get model weight list
        :return: List of model weights
        """
        raise NotImplementedError

