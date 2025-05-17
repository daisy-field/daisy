# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""A collection of interfaces and base classes for internal components for federated
learning, from the models who are trained on and process data points in online-manner
to the aggregators that must be able to aggregate generic models into a singular one,
all of which are compatible with the federated components in the other sub-package.

Currently, the following generic model classes are supported:

    * FederatedModel - Interface class. Any model that is provided to the federated
    system must implement this.
    * TFFederatedModel - Generic class wrapper for generic tensorflow models for FL.
        * get_fae() - Factory method to create a sample federated autoencoder model of
        a fixed depth but with variable input size.
    * FederatedIFTM - IFTM (i.e., hybrid) model class for federated anomaly detection.

For the IFTM model classes, there is also a set of base-case threshold models (TMs)
provided:

    * FederatedTM - Interface class for threshold models used in IFTM and for other
    hybrid detection approaches.
    * AvgTM - Extended interface class for average(+std.)-based threshold models.
    * CumAvgTM - Cumulative (online) averaging for threshold model computation.
    * SMAvgTM - Simple moving averaging for threshold model computation.
    * EMAvgTM - Exponential moving averaging for threshold model computation.
    * MadTM - Median absolute deviation-based threshold models.

For the aggregators, the following structure of interfaces and classes is provided:

    * Aggregator - Interface class from which all aggregators implement/extend from,
    able to handle any parameters.
    * ModelAggregator - Extended aggregator interface class, solely for the
    aggregation of model parameters.
    * FedAvgAggregator - Model aggregator following FedAvg algorithm --- simply
    computing batch average of models.
    * CumAggregator - Cumulative online average for model aggregation.
    * SMAggregator - Simple moving (sliding window) average for model aggregation.
    * EMAggregator - Exponential moving average for model aggregation.

Author: Fabian Hofmann, Seraphin Zunzer
Modified: 04.04.24
"""

__all__ = [
    "FederatedModel",
    "TFFederatedModel",
    "FederatedIFTM",
    "FederatedTM",
    "AvgTM",
    "CumAvgTM",
    "SMAvgTM",
    "EMAvgTM",
    "MadTM",
    "Aggregator",
    "ModelAggregator",
    "FedAvgAggregator",
    "CumAggregator",
    "SMAggregator",
    "EMAggregator",
    "ThresholdModelSimon",
    "FixThreshold",
    "EMAMADThresholdModel",
]

from .federated_aggregator import Aggregator, ModelAggregator
from .federated_aggregator import CumAggregator, SMAggregator, EMAggregator
from .federated_aggregator import FedAvgAggregator
from .federated_model import FederatedIFTM
from .federated_model import FederatedModel, TFFederatedModel
from .threshold_models import AvgTM, CumAvgTM, SMAvgTM, EMAvgTM
from .threshold_models import FederatedTM
from .threshold_models import MadTM
from .threshold_models import ThresholdModelSimon, FixThreshold, EMAMADThresholdModel
