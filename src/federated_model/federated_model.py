"""
    Interface for federated models.

    Author: Seraphin Zunzer
    Modified: 09.08.23
"""
from abc import ABC, abstractmethod


class FederatedModel(ABC):

    @abstractmethod
    def get_model_weights(self):
        """
        Function to get model weight list
        :return: List of model weights
        """
        raise NotImplementedError

    @abstractmethod
    def set_model_weights(self, weights: []):
        """
        Function to set model weights
        :return: -
        """
        raise NotImplementedError

    @abstractmethod
    def build_model(self):
        """
        Function to bild federated model
        :return: built model
        """
        raise NotImplementedError

    @abstractmethod
    def compile_model(self):
        """
        Function to compile a model for prediction
        :return: compiled model
        """
        raise NotImplementedError

    @abstractmethod
    def fit_model(self, **kwargs):
        """
        Function to compile a model for prediction
        :return: compiled model
        """
        raise NotImplementedError

    @abstractmethod
    def model_predict(self, **kwargs):
        """
        Function to compile a model for prediction
        :return: compiled model
        """
        raise NotImplementedError
