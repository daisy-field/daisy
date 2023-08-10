"""
    Online federated average method.

    Author: Seraphin Zunzer
    Modified: 09.08.23
"""


import keras
import tensorflow as tf
from model_aggregation.federated_aggregation import FederatedAggregation
import logging

input_size = 70


class FedAvg(FederatedAggregation):
    """Class for federated average aggregation"""

    def aggregate(self, global_weights, client_weights):
        """
        Calculate fedavg between global weights and new weights.
        When weights are malformed, skip aggregation and return old weights.

        :param global_weights: Weights of global model
        :param client_weights: Weights of client model that should be aggregated
        :return: New weights for global model
        """
        old_weights = global_weights
        try:
            for i in range(0, len(global_weights)):
                for j in range(0, len(global_weights[i])):
                    global_weights[i][j] = (global_weights[i][j] + client_weights[i][j]) / 2

            return global_weights
        except:
            logging.warning("Malformed weights received!")
            return old_weights