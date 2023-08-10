"""
    Interface for federated models.

    Author: Seraphin Zunzer
    Modified: 09.08.23
"""
from abc import ABC, abstractmethod


class FederatedModel(ABC):
    model = None

    @abstractmethod
    def compile_model(self):
        """
        Function to compile a model for prediction
        :return: compiled model
        """
        raise NotImplementedError

