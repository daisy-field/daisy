# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Collection of the abstract and base classes that represent the actual core-components of the federated intrusion
detection system, from the nodes monitoring and processing the distributed datastreams, to the aggregation servers
to which models, events, and evaluation results are reported.

For the processing nodes encapsulating the actual federated models and that learn and process data points in online-
manner, the following classes are supported:

    * FederatedOnlineNode - Abstract federated node that learns on generic streaming data.
    * FederatedOnlineClient - Federated node that handles model aggregation via a centralized aggregation server.
    * FederatedOnlinePeer - Base class for federated nodes that feature master-/server-less aggregation strategies.

For the aggregation servers, the following interfaces and implementations is provided:

    * FederatedOnlineAggregator - Abstract aggregators that perform aggregation at runtime, continuously.
    * FederatedModelAggregator - Base class for client-server-based model aggregation, i.e. counterpart to the
    FederatedOnlineClient federate online node implementation.
    * FederatedValueAggregator - Base class for generic value aggregation via simple value caching using a sliding
    window for each federated node.
    * FederatedPredictionAggregator - Aggregator for prediction values from federated IDS nodes. Dashboard enabled.
    * FederatedEvaluationAggregator - Aggregator for evaluation values from federated nodes. Dashboard enabled.

Author: Fabian Hofmann, Seraphin Zunzer
Modified: 27.11.23
"""

from .aggregator import FederatedModelAggregator
from .aggregator import (
    FederatedValueAggregator,
    FederatedPredictionAggregator,
    FederatedEvaluationAggregator,
)
from .aggregator import FederatedOnlineAggregator

# from .evaluator import FederatedOnlineEvaluator
from .node import FederatedOnlineClient, FederatedOnlinePeer
from .node import FederatedOnlineNode
