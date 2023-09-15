"""
    A collection of interfaces and base classes for internal components for federated learning, from the models who are
    trained on and process data points in online-manner to the aggregators that must be able to aggregate generic models
    into a singular one, all of which are compatible with the federated components in the other sub-package.

    Currently, the following generic model classes are supported:

        * FederatedModel - Interface class. Any model that is provided to the federated system must implement this.
        * TFFederatedModel - Generic class wrapper for generic tensorflow models for federated learning.
        * FederatedIFTM - IFTM (i.e., hybrid) model class for federated anomaly detection.

    For the IFTM model classes, there are also a set of base-case threshold models (TMs) provided:

        * FederatedTM - TODO

    For the aggregators, the following structure of interfaces and classes is provided:

        * Aggregator - Interface class from which all aggregators implement/extend from, able to handle any parameters.
        * ModelAggregator - Extended aggregator interface class, solely for the aggregation of model parameters.
        * FedAvgAggregator - Model aggregator following FedAvg algorithm --- simply computing batch average of models.
        * CumAggregator - Cumulative online average for model parameter aggregation.
        * SMAggregator - Simple moving (sliding window) average for model parameter aggregation.
        * EMAggregator - Exponential moving average for model parameter aggregation.

    Author: Fabian Hofmann, Seraphin Zunzer
    Modified: 31.08.23
"""

from .federated_aggregator import Aggregator, ModelAggregator
from .federated_aggregator import CumAggregator, SMAggregator, EMAggregator
from .federated_aggregator import FedAvgAggregator
from .federated_model import FederatedIFTM
from .federated_model import FederatedModel
from .federated_model import TFFederatedModel
