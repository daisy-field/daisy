# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""A collection of various types of federated aggregators, implementing the same
interface for each federated aggregator type. Each of them receives and aggregates
data from federated nodes at runtime in a client-server-based exchange, continuously,
optionally forwarding the data, if of interest, to an additional dashboard,
which follows a simple REST API (see the dashboard subpackage).

Author: Fabian Hofmann, Seraphin Zunzer
Modified: 04.04.24
"""
# TODO Future Work: Defining granularity of logging in inits

import logging
import threading
from abc import ABC, abstractmethod
from collections import deque
from random import sample
from time import sleep
from typing import cast, Sequence

import numpy as np
import requests

from daisy.communication import EndpointServer, StreamEndpoint
from daisy.federated_learning import ModelAggregator


class FederatedOnlineAggregator(ABC):
    """Abstract class for generic federated online aggregators, that receive and
    aggregates data from federated nodes at runtime in a client-server-based
    exchange, continuously. To accomplish this, the abstract class in its core is
    merely a wrapper around the EndpointServer class to process new incoming clients,
    leaving the actual aggregation functionality to its implementations, which run in
    threaded manner.

    To realize this, the following methods must be implemented:

        * setup(): Setup function for any state variables called during start.

        * cleanup(): Cleanup function for any state variables called during stopping.

        * create_fed_aggr(): Encapsulates the aggregation loop for the entire life-cycle
        of the aggregator.

    Note that an optionally running dashboard server (see the dashboard subpackage),
    can be used to collect and display aggregated values further. To implement this,
    _update_dashboard() can be used during the aggregation loop.
    """

    _logger: logging.Logger

    _aggr_serv: EndpointServer
    _timeout: int

    _dashboard_url: str

    _fed_aggr: threading.Thread
    _started: bool

    def __init__(
        self,
        addr: tuple[str, int],
        name: str = "",
        timeout: int = 10,
        dashboard_url: str = None,
    ):
        """Creates a new federated online aggregator.

        :param addr: Address of aggregation server for federated nodes to report to.
        :param name: Name of federated online aggregator for logging purposes.
        :param timeout: Timeout for waiting to receive message from federated nodes.
        :param dashboard_url: URL to dashboard to report aggregated data to, depending
        on class implementations (see class docstring).
        """
        self._logger = logging.getLogger(name)
        self._logger.info("Initializing federated online aggregator...")

        self._aggr_serv = EndpointServer(
            name="Server", addr=addr, c_timeout=timeout, multithreading=True
        )
        self._timeout = timeout

        self._dashboard_url = dashboard_url

        self._started = False
        self._logger.info("Federated online aggregator initialized.")

    def start(self):
        """Starts the federated online aggregator, along with its underlying server,
        and any other object by an extension of this class (see setup()). Non-blocking.

        :raises RuntimeError: If federated online aggregator has already been started.
        """
        self._logger.info("Starting federated online aggregator...")
        if self._started:
            raise RuntimeError("Federated online aggregator has already been started!")
        self._started = True
        try:
            self._aggr_serv.start()
        except RuntimeError:
            pass
        self._logger.info("Performing further setup...")
        self.setup()

        self._fed_aggr = threading.Thread(target=self.create_fed_aggr, daemon=True)
        self._fed_aggr.start()
        self._logger.info("Federated online aggregator started.")

    @abstractmethod
    def setup(self):
        """Setup function that must be implemented, called during start(); sets any
        new internal state variables and objects up used during the aggregation process.

        Note that any such instance attribute should be initialized within an
        extension of the __init_() method.
        """
        raise NotImplementedError

    def stop(self):
        """Stops the federated online aggregator, along with its underlying server,
        and any other object by an extension of this class (see cleanup()).

        :raises RuntimeError: If federated online aggregator has not been started.
        """
        self._logger.info("Stopping federated online aggregator...")
        if not self._started:
            raise RuntimeError("Federated online aggregator has not been started!")
        self._started = False
        try:
            self._aggr_serv.stop(timeout=self._timeout)
        except RuntimeError:
            pass
        self._logger.info("Performing further cleanup...")
        self.cleanup()

        self._fed_aggr.join()
        self._logger.info("Federated online aggregator stopped.")

    @abstractmethod
    def cleanup(self):
        """Cleanup function that must be implemented, called during stop(); resets
        any new internal state variables and objects up used during the federated
        updating process, both for synchronous and asynchronous learning.
        """
        raise NotImplementedError

    @abstractmethod
    def create_fed_aggr(self):
        """Continuous, asynchronous federated aggregation loop, that runs over the
        entire life-cycle. Must use the _started semaphore to exit the loop in case
        the node is stopped (see stop()).

        Since the abstract federated online aggregator does not define which kind of
        data is aggregated, the aggregation can be equally manifold. The only
        property shared is the usage of the _aggr_serv endpoint server,
        to communicate with federated nodes.
        """
        raise NotImplementedError

    def _update_dashboard(self, ressource: str, data: dict):
        """Updates a specific ressource of the dashboard, if given and available with
        given data. Note that both the available resources and the to be reported
        data is determined by the dashboard's endpoints, and must be
        adjusted/extended if more data is to be reported than current functionality
        allows (see dashboard subpackage)

        :param ressource: URL-suffix to ressource.
        :param data: Data to report.
        """
        if self._dashboard_url is None:
            return
        try:
            _ = requests.post(
                url="http://" + self._dashboard_url + ":8000" + ressource, data=data
            )
        except requests.exceptions.RequestException as e:
            self._logger.warning(f"Dashboard server not reachable: {e}")

    def __del__(self):
        if self._started:
            self.stop()


class FederatedModelAggregator(FederatedOnlineAggregator):
    """Centralized federated learning is the simplest federated approach, since any
    client in the topology, while learning by itself, always reports to the same
    centralized server that aggregate the models in their stead. To accomplish this,
    this implementation additionally wraps itself around the model aggregator.

    Thus, this implementation is essentially the inverse counterpart to
    FederatedOnlineClient, which either operates in synchronous fashion --- polling a
    sample of the existing clients to receive their models, aggregating them,
    before sending the new global model to the entire population, or in asynchronous
    mode. The latter behaves inverse to the federated client, since it periodically
    checks the connected clients for message activity, aggregating any received
    models with the current global one and sending them back to the respective client.
    Note that the latter only works if the model aggregator has an internal state
    for the global model.
    """

    _m_aggr: ModelAggregator

    _update_interval: int
    _num_clients: int
    _min_clients: float

    def __init__(
        self,
        m_aggr: ModelAggregator,
        addr: tuple[str, int],
        name: str = "",
        timeout: int = 10,
        update_interval: int = None,
        num_clients: int = None,
        min_clients: float = 0.5,
        dashboard_url: str = None,
    ):
        """Creates a new federated model aggregator. If update_interval is not set,
        defaults to asynchronous federated model aggregation, i.e. waiting for
        individual clients to report their local models in unspecified/unset
        intervals, before sending them the freshly aggregated global model back.

        :param m_aggr: Actual aggregator to aggregate models with.
        :param addr: Address of aggregation server for clients to connect to.
        :param name: Name of federated model aggregator for logging purposes.
        :param timeout: Timeout for waiting to receive local model updates from
        federated clients.
        :param update_interval: Sets how often the aggregation server should do a (
        synchronous) federated aggregation step to compute a global model to send to
        the clients (in seconds).
        :param num_clients: Allows sampling of client population to perform a sampled
        synchronous aggregation step. If none provided, always attempts to request
        models from the entire population. For async mode, threshold of waiting
        clients for a model update.
        :param min_clients: Minimum ratio of responsive clients after a request for
        models during a sampled synchronous aggregation step, aborts aggr step if not
        satisfied. If none provided, tolerates a 50% failure rate.
        :param dashboard_url: URL to dashboard to report aggregation statics to.
        """
        super().__init__(
            addr=addr, name=name, timeout=timeout, dashboard_url=dashboard_url
        )

        self._m_aggr = m_aggr

        self._update_interval = update_interval
        self._num_clients = num_clients
        self._min_clients = min_clients

    def setup(self):
        pass

    def cleanup(self):
        pass

    def create_fed_aggr(self):
        """Starts the loop to either synchronously aggregate in fixed intervals or
        check in looping manner whether there are federated clients requesting an
        asynchronous updating step.
        """
        self._logger.info("Starting model aggregation loop...")
        while self._started:
            try:
                if self._update_interval is not None:
                    self._logger.debug(
                        "Initiating interval-based synchronous aggregation step..."
                    )
                    self._sync_aggr()
                    sleep(self._update_interval)
                else:
                    self._logger.debug(
                        "Checking federated clients for "
                        "asynchronous aggregation requests..."
                    )
                    self._async_aggr()
                    # FIXME this is getting spammed if there are no read ready clients
            except RuntimeError:
                # stop() was called
                break
        self._logger.info("Model aggregation loop stopped.")

    def _sync_aggr(self):
        """Performs a synchronous, sampled federated aggregation step, i.e. sending
        out a request for local models to a sample of the client population. If
        sufficient clients respond to the request after a set timeout, their models
        are aggregated, potentially also taking into account the global model of the
        previous step (depending on the model aggregator chosen), and the newly
        created global model is sent back to all available clients.
        """
        clients = self._aggr_serv.poll_connections()[1].values()
        if self._num_clients is not None:
            if len(clients) < self._num_clients:
                self._logger.info(
                    f"Insufficient write-ready clients [{len(clients)}] "
                    f"available for aggregation step [{self._num_clients}]!"
                )
                return
            clients = sample(cast(Sequence, clients), self._num_clients)
        num_clients = len(clients)

        self._logger.debug(
            f"Requesting local models from sampled clients [{len(clients)}]..."
        )
        for client in clients:
            client.send(None)
        sleep(self._timeout)

        clients = StreamEndpoint.select_eps(clients)[0]
        self._logger.debug(
            "Receiving local models from available "
            f"requested clients [{len(clients)}]..."
        )
        client_models = [
            model
            for model in cast(
                list[list[np.ndarray]],
                StreamEndpoint.receive_latest_ep_objs(clients, list).values(),
            )
            if model is not None
        ]

        if len(client_models) < min(1, int(num_clients * self._min_clients)):
            self._logger.info(
                f"Insufficient number of client models [{len(client_models)}] "
                f"for aggregation received [{self._num_clients}]!"
            )
            return
        self._logger.debug(
            f"Aggregating client models [{len(client_models)}] into global model..."
        )
        global_model = self._m_aggr.aggregate(client_models)

        clients = self._aggr_serv.poll_connections()[1].values()
        self._logger.debug(
            f"Sending aggregated global model to all "
            f"available clients [{len(clients)}]..."
        )
        for client in clients:
            client.send(global_model)
        self._logger.info("Sending aggregated global model to dashboard ")

        self._update_dashboard(
            "/aggregation/",
            {
                "agg_status": "Operational",  # TODO add len(client_models)
                "agg_count": len(self._aggr_serv.poll_connections()[1].values()),
                "agg_nodes": str(self._aggr_serv.poll_connections()[1]),
            },
        )

    def _async_aggr(self):
        """Performs an asynchronous federated aggregation step, i.e. checking whether
        a sufficient number of clients sent their local models to the server,
        before aggregating their models, potentially also taking into account the
        global model of the previous step (depending on the model aggregator chosen).
        The newly created global model is sent back only to all clients that
        requested an aggregation step.
        """
        clients = self._aggr_serv.poll_connections()[0].values()
        if self._num_clients is not None and len(clients) < self._num_clients:
            self._logger.info(
                f"Insufficient read-ready clients [{len(clients)}] "
                f"available for aggregation step [{self._num_clients}]!"
            )
            sleep(1)
            return

        self._logger.debug(
            f"Receiving local models from requesting clients [{len(clients)}]..."
        )
        client_models = [
            model
            for model in cast(
                list[list[np.ndarray]],
                StreamEndpoint.receive_latest_ep_objs(clients, list).values(),
            )
            if model is not None
        ]

        if (
            len(client_models) == 0
            or self._num_clients is not None
            and len(client_models) < self._num_clients
        ):
            self._logger.info(
                f"Insufficient number of client models [{len(client_models)}] "
                f"for aggregation received [{self._num_clients}]!"
            )
            sleep(1)
            return
        self._logger.debug(
            f"Aggregating client models [{len(client_models)}] into global model..."
        )
        global_model = self._m_aggr.aggregate(client_models)

        clients = StreamEndpoint.select_eps(clients)[1]
        self._logger.debug(
            "Sending aggregated global model to available "
            f"requesting clients [{len(clients)}]..."
        )
        for client in clients:
            client.send(global_model)

        self._update_dashboard(
            "/aggregation/",
            {
                "agg_status": "Operational",  # TODO add len(client_models)
                "agg_count": len(clients),
                "agg_nodes": clients,
            },
        )


class FederatedValueAggregator(FederatedOnlineAggregator):
    """Base class for generic value aggregation of messages which are sent
    continuously by the federated nodes to the aggregation server. These values are
    treated as some sort of series, each assigned to a federated node and its
    respective endpoint from which new values can be retrieved and stored in an
    individual sliding window queue.

    Since the content of messages sent by the federated nodes are shaped in various
    ways, depending on their type (potentially containing multiple values even),
    extensions of this class may be required, which should at least override the
    following methods:

        * process_node_msg(): Converts a singular received message from a federated
        node into a list of values to be added to the sliding window queue of that
        node. Per default this method assumes a singleton value. If a dashboard is
        supported, should also call _update_dashboard() to report aggregated values
        of interest.

    Note that the base class could be extended in various other ways as well,
    it is recommended to put such functionality into the setup() and cleanup() methods.
    """

    _aggr_values: dict[tuple[str, int], deque]
    _window_size: int

    def __init__(
        self,
        addr: tuple[str, int],
        name: str = "",
        timeout: int = 10,
        window_size: int = None,
        dashboard_url: str = None,
    ):
        """Creates a new federated value aggregator.

        :param addr: Address of aggregation server for federated nodes to report to.
        :param name: Name of federated value aggregator for logging purposes.
        :param timeout: Timeout for waiting to receive message from federated nodes.
        :param window_size: Maximum number of latest received entries stored for each
        federated node.
        :param dashboard_url: URL to dashboard to report aggregated values (potentially
        filtered by interest) to.
        """
        super().__init__(
            addr=addr, name=name, timeout=timeout, dashboard_url=dashboard_url
        )

        self._window_size = window_size

    def setup(self):
        self._aggr_values = {}

    def cleanup(self):
        self._aggr_values = {}

    def create_fed_aggr(self):
        """Starts the loop to continuously poll the federated nodes for new values to
        receive and process, before adding them to the datastructure.
        """
        self._logger.info("Starting result aggregation loop...")
        while self._started:
            try:
                nodes = self._aggr_serv.poll_connections()[0].items()
                if len(nodes) == 0:
                    sleep(self._timeout)
                    continue

                for node, node_ep in nodes:
                    if node not in self._aggr_values:
                        self._aggr_values[node] = deque(maxlen=self._window_size)
                    try:
                        while True:
                            new_values = self.process_node_msg(
                                node, node_ep.receive(timeout=0)
                            )
                            self._aggr_values[node].extend(new_values)
                    except (RuntimeError, TimeoutError):
                        pass
            except RuntimeError:
                # stop() was called
                break
        self._logger.info("Result aggregation loop stopped.")

    def process_node_msg(self, node: tuple[str, int], msg) -> list:
        """Converts a singular received message from a federated node into a list of
        values to be added to the sliding window queue of that client. Per default
        this method assumes a singleton value and does not anything to a potentially
        running dashboard (see class docstring for extending the base class).

        :param node: Node from whom the message was received.
        :param msg: Message to be processed.
        :return: List of values to be added to the sliding window of the respective
        node.
        """
        self._logger.debug(f"Value received from {node}: {msg}")
        return [msg]


class FederatedPredictionAggregator(FederatedValueAggregator):
    """Aggregator for prediction values from a federated IDS. Since federated IDS
    nodes report their predictions in minibatches, each message received from a node
    contains a multitude of values, which must be fragmented first, before they can
    be stored.

    Note that for the purposes of the dashboard, these predictions are filtered, with
    only alerts being forwarded to the other component.
    """

    def __init__(
        self,
        addr: tuple[str, int],
        name: str = "",
        timeout: int = 10,
        window_size: int = None,
        dashboard_url: str = None,
    ):
        """Creates a new federated prediction aggregator.

        :param addr: Address of aggregation server for federated nodes to report to.
        :param name: Name of federated value aggregator for logging purposes.
        :param timeout: Timeout for waiting to receive message from federated nodes.
        :param window_size: Maximum number of latest received entries stored for each
        federated node.
        :param dashboard_url: URL to dashboard to report federated IDS alerts to.
        """
        super().__init__(
            addr=addr,
            name=name,
            timeout=timeout,
            window_size=window_size,
            dashboard_url=dashboard_url,
        )

    def create_fed_aggr(self):
        """Starts the loop to continuously poll the federated nodes for new values to
        receive and process, before adding them to the datastructure.
        """
        self._logger.info("Starting result aggregation loop...")
        while self._started:
            # TODO @seraphin report heartbeat (agg status?), incl. active connections
            # TODO add client number
            self._update_dashboard(
                "/prediction/",  # ADD CORRECT PATH
                {
                    "pred_status": "Operational",
                    "pred_count": len(self._aggr_serv.poll_connections()[1].values()),
                    "pred_nodes": str(self._aggr_serv.poll_connections()[1]),
                },  # TODO get correct value for pred_count
            )

            try:
                nodes = self._aggr_serv.poll_connections()[0].items()
                if len(nodes) == 0:
                    sleep(self._timeout)
                    continue

                for node, node_ep in nodes:
                    if node not in self._aggr_values:
                        self._aggr_values[node] = deque(maxlen=self._window_size)
                    try:
                        while True:
                            new_values = self.process_node_msg(
                                node, node_ep.receive(timeout=0)
                            )
                            self._aggr_values[node].extend(new_values)
                    except (RuntimeError, TimeoutError):
                        pass
            except RuntimeError:
                # stop() was called
                break
        self._logger.info("Result aggregation loop stopped.")

    def process_node_msg(
        self, node: tuple[str, int], msg: tuple[np.ndarray, np.ndarray]
    ) -> list[tuple]:
        """Converts a received message from a federated node containing a minibatch
        of predictions into a list of tuples that each contain a datapoint and its
        respective prediction. If an IDS alert is identified, forwards it to the
        dashboard, if available.

        :param node: Node from whom the message was received.
        :param msg: Minibatch arranged in x,y fashion to be disassembled into value
        pairs.
        :return: List of value pairs (x,f(x)) to be added to the sliding window of
        the respective node.
        """
        values = []
        x_data, y_pred = msg
        for i in range(len(x_data)):
            t = x_data[i], y_pred[i]
            self._logger.debug(f"Prediction received from {node}: {t}")
            values.append(t)

            # TODO @seraphin adjust values to report (see blow)
            if y_pred[i]:
                self._update_dashboard(
                    "/alert/",
                    {
                        "address": str(node[0])
                        + ":"
                        + str(node[1]),  # ADD NODE WHO REPORTED ALERT
                        "category": "alert",
                        "active": True,
                        "message": "Packet content: "
                        + ",".join(str(x) for x in x_data[i]),  # "Alert raised!",
                    },
                )
        return values


class FederatedEvaluationAggregator(FederatedValueAggregator):
    """Aggregator for evaluation metric values from a federated IDS. Since these values
    depend on the evaluation metrics chosen during the instantiation of a federated
    IDS node, a proper disassembly and aggregation of any values is currently not
    implemented and requires an additional extension of this class (for example
    computing metric averages over all federated nodes). Instead, specific evaluation
    metric values of the ConfMatrSlidingWindowEvaluation metric class (see the
    evaluation subpackage) are directly forwarded to the dashboard to be displayed,
    if they are used by a reporting IDS node.
    """

    def __init__(
        self,
        addr: tuple[str, int],
        name: str = "",
        timeout: int = 10,
        window_size: int = None,
        dashboard_url: str = None,
    ):
        """Creates a new federated evaluation aggregator.

        :param addr: Address of aggregation server for federated nodes to report to.
        :param name: Name of federated value aggregator for logging purposes.
        :param timeout: Timeout for waiting to receive message from federated nodes.
        :param window_size: Maximum number of latest received entries stored for each
        federated node.
        :param dashboard_url: URL to dashboard to report evaluation metric values to.
        """
        super().__init__(
            addr=addr,
            name=name,
            timeout=timeout,
            window_size=window_size,
            dashboard_url=dashboard_url,
        )

    def create_fed_aggr(self):
        """Starts the loop to continuously poll the federated nodes for new values to
        receive and process, before adding them to the datastructure.
        """
        self._logger.info("Starting result aggregation loop...")
        while self._started:
            self._update_dashboard(
                "/evaluation/",  # ADD CORRECT PATH
                {
                    "eval_status": "Operational",
                    "eval_count": len(self._aggr_serv.poll_connections()[1].values()),
                    "eval_nodes": str(self._aggr_serv.poll_connections()[1]),
                },  # TODO get correct value for eval_count
            )

            try:
                nodes = self._aggr_serv.poll_connections()[0].items()
                if len(nodes) == 0:
                    sleep(self._timeout)
                    continue

                for node, node_ep in nodes:
                    if node not in self._aggr_values:
                        self._aggr_values[node] = deque(maxlen=self._window_size)
                    try:
                        while True:
                            new_values = self.process_node_msg(
                                node, node_ep.receive(timeout=0)
                            )
                            self._aggr_values[node].extend(new_values)
                    except (RuntimeError, TimeoutError):
                        pass
            except RuntimeError:
                # stop() was called
                break
        self._logger.info("Result aggregation loop stopped.")

    def process_node_msg(self, node: tuple[str, int], msg):
        """Converts a received message from a federated node containing the current
        values of its evaluation metrics after processing a minibatch. Currently, not
        fully implemented, as it simply echoes the message to be appended to the
        sliding window queue. However, if ConfMatrSlidingWindowEvaluation metric
        values are identified, forwards a subset of them to the dashboard, if available.

        :param node: Node from whom the message was received.
        :param msg: Evaluation metrics as a dictionary of metrics and their values.
        :return: Message as received.
        """
        self._logger.debug(f"Evaluation metrics received from {node}: {msg}")

        if "conf_matrix_online_evaluation" in msg:
            conf_matrix = msg["conf_matrix_online_evaluation"]
            self._update_dashboard(
                "/metrics/",
                {
                    "address": node,
                    "accuracy": conf_matrix["accuracy"],
                    "recall": conf_matrix["recall"],
                    "precision": conf_matrix["precision"],
                    "f1": conf_matrix["f1 measure"],
                },
            )
        return msg
