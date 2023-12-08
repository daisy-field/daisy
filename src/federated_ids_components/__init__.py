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
        * FederatedResultAggregator -
        * FederatedEvalAggregator -

    Author: Fabian Hofmann, Seraphin Zunzer
    Modified: 27.11.23

    FIXME FEDERATED_ONLINE_PEER + FEDERATED_RESULT_AGGR + FEDERATED_EVAL_AGGR
"""

from .aggregator import FederatedModelAggregator, FederatedResultAggregator
from .aggregator import FederatedOnlineAggregator
from .evaluator import FederatedOnlineEvaluator
from .node import FederatedOnlineClient, FederatedOnlinePeer
from .node import FederatedOnlineNode
