"""
    A collection of various types of federated aggregators, implementing the same interface for each federated
    aggregator type. Each of them receives and aggregates data from federated nodes at runtime in a client-server-based
    exchange, continuously.

    Author: Fabian Hofmann
    Modified: 22.01.24

    TODO: Adapt ResultAggregator to needs
    TODO: - Integrate Evaluator into ValueAggregator (i.e. inherit from it)
    TODO Future Work: Defining granularity of logging in inits
"""
import logging
import threading
from abc import ABC, abstractmethod
from collections import deque
from random import sample
from time import sleep
from typing import cast, Sequence

import numpy as np

from daisy.communication import EndpointServer, ep_select, receive_latest_ep_objs
from daisy.federated_learning import ModelAggregator


class FederatedOnlineAggregator(ABC):
    """Abstract class for generic federated online aggregators, that receive and aggregates data from federated nodes at
    runtime in a client-server-based exchange, continuously. To accomplish this, the abstract class in its core is
    merely a wrapper around the EndpointServer class to process new incoming clients, leaving the actual aggregation
    functionality to its implementations, which run in threaded manner.

    To realize this, the following methods must be implemented:

        * setup(): Setup function for any state variables called during the starting of the starting of the aggregator.

        * cleanup(): Cleanup function for any state variables called during the stopping of the aggregator.

        * create_fed_aggr(): Encapsulates the aggregation loop for the entire life-cycle of the aggregator.
    """
    _logger: logging.Logger

    _aggr_serv: EndpointServer
    _timeout: int

    _fed_aggr: threading.Thread
    _started: bool

    def __init__(self, addr: tuple[str, int], name: str = "", timeout: int = 10):
        """Creates a new federated online aggregator.

        :param addr: Address of aggregation server for federated nodes to report to.
        :param name: Name of federated online aggregator for logging purposes.
        :param timeout: Timeout for waiting to receive message from federated nodes.
        """
        self._logger = logging.getLogger(name)
        self._logger.info("Initializing federated online aggregator...")

        self._aggr_serv = EndpointServer(name="Server", addr=addr, c_timeout=timeout, multithreading=True)
        self._timeout = timeout

        self._started = False
        self._logger.info("Federated online aggregator initialized.")

    def start(self):
        """Starts the federated online aggregator, along with its underlying server, and any other object by an
        extension of this class (see setup()). Non-blocking.

        :raises RuntimeError: If federated online aggregator has already been started.
        """
        self._logger.info("Starting federated online aggregator...")
        if self._started:
            raise RuntimeError(f"Federated online aggregator has already been started!")
        self._started = True
        try:
            self._aggr_serv.start(),
        except RuntimeError:
            pass
        self._logger.info("Performing further setup...")
        self.setup()

        self._fed_aggr = threading.Thread(target=self.create_fed_aggr(), daemon=True)
        self._fed_aggr.start()
        self._logger.info("Federated online aggregator started.")

    @abstractmethod
    def setup(self):
        """Setup function that must be implemented, called during start(); sets any new internal state variables and
        objects up used during the aggregation process.

        Note that any such instance attribute should be initialized within an extension of the __init_() method.
        """
        raise NotImplementedError

    def stop(self):
        """Stops the federated online aggregator, along with its underlying server, and any other object by an extension
        of this class (see cleanup()).

        :raises RuntimeError: If federated online aggregator has not been started.
        """
        self._logger.info("Stopping federated online aggregator...")
        if not self._started:
            raise RuntimeError(f"Federated online aggregator has not been started!")
        self._started = False
        try:
            self._aggr_serv.stop(timeout=self._timeout),
        except RuntimeError:
            pass
        self._logger.info("Performing further cleanup...")
        self.cleanup()

        self._fed_aggr.join()
        self._logger.info("Federated online aggregator stopped.")

    @abstractmethod
    def cleanup(self):
        """Cleanup function that must be implemented, called during stop(); resets any new internal state variables and
        objects up used during the federated updating process, both for synchronous and asynchronous learning.
        """
        raise NotImplementedError

    @abstractmethod
    def create_fed_aggr(self):
        """Continuous, asynchronous federated aggregation loop, that runs over the entire life-cycle. Must use the
        _started semaphore to exit the loop in case the node is stopped (see stop()).

        Since the abstract federated online aggregator does not define which kind of data is aggregated, the aggregation
        can be equally manifold. The only property shared is the usage of the _aggr_serv endpoint server, to communicate
        with federated nodes.
        """
        raise NotImplementedError

    def __del__(self):
        if self._started:
            self.stop()


class FederatedModelAggregator(FederatedOnlineAggregator):
    """Centralized federated learning is the simplest federated approach, since any client in the topology, while
    learning by itself, always reports to the same centralized server that aggregate the models in their stead. To
    accomplish this, this implementation additionally wraps itself around the model aggregator.

    Thus, this implementation is essentially the inverse counterpart to FederatedOnlineClient, which either operates in
    synchronous fashion --- polling a sample of the existing clients to receive their models, aggregating them, before
    sending the new global model to the entire population, or in asynchronous mode. The latter behaves inverse to the
    federated client, since it periodically checks the connected clients for message activity, aggregating any received
    models with the current global one and sending them back to the respective client. Note that the latter only works
    if the model aggregator has an internal state for the global model.
    """
    _m_aggr: ModelAggregator

    _online_update: bool

    _update_interval_t: int
    _num_clients: int
    _min_clients: float

    def __init__(self, m_aggr: ModelAggregator, addr: tuple[str, int], name: str = "",
                 timeout: int = 10, online_update: bool = True,
                 update_interval_t: int = None, num_clients: int = None, min_clients: float = 0.5):
        """Creates a new federated model aggregator.

        :param m_aggr: Actual aggregator to aggregate models with.
        :param addr: Address of aggregation server for clients to connect to.
        :param name: Name of federated model aggregator for logging purposes.
        :param timeout: Timeout for waiting to receive local model updates from federated clients.
        :param online_update: Async updating, i.e. waiting for individual clients to report their local models in
        unspecified/unset intervals, before sending them the freshly aggregated global model back. Enabled by default.
        :param update_interval_t: If async disabled, sets how often the aggregation server should do a federated
        aggregation step to compute a global model to send to the clients (in seconds).
        :param num_clients: Allows sampling of client population to perform a sampled synchronous aggregation step. If
        none provided, always attempts to request models from the entire population.
        :param min_clients: Minimum ratio of responsive clients to requested ones, aborts aggr step if not satisfied. If
        none provided, tolerates a 50% failure rate.
        """
        super().__init__(addr=addr, name=name, timeout=timeout)

        self._m_aggr = m_aggr

        self._online_update = online_update

        self._update_interval_t = update_interval_t
        self._num_clients = num_clients
        self._min_clients = min_clients

    def setup(self):
        pass

    def cleanup(self):
        pass

    def create_fed_aggr(self):
        """Starts the loop to either synchronously aggregate in fixed intervals or check in looping manner whether there
        are federated clients requesting an asynchronous updating step.
        """
        self._logger.info("Starting model aggregation loop...")
        while self._started:
            try:
                if self._update_interval_t is not None:
                    self._logger.debug("Initiating interval-based synchronous aggregation step...")
                    self._sync_aggr()
                    sleep(self._update_interval_t)

                elif self._online_update:
                    self._logger.debug("Checking federated clients for asynchronous aggregation requests...")
                    self._async_aggr()

                else:
                    # Federated aggregation disabled
                    break
            except RuntimeError:
                # stop() was called
                break
        self._logger.info("Model aggregation loop stopped.")

    def _sync_aggr(self):
        """Performs a synchronous, sampled federated aggregation step, i.e. sending out a request for local models to a
        sample of the client population. If sufficient clients respond to the request after a set timeout, their models
        are aggregated, potentially also taking into account the global model of the previous step (depending on the
        model aggregator chosen), and the newly created global model is sent back to all available clients.
        """
        clients = self._aggr_serv.poll_connections()[1].values()
        if self._num_clients is not None:
            if len(clients) < self._num_clients:
                self._logger.info(f"Insufficient write-ready clients available for aggregation step [{len(clients)}]!")
                return
            clients = sample(cast(Sequence, clients), self._num_clients)
        num_clients = len(clients)

        self._logger.debug(f"Requesting local models from sampled clients [{len(clients)}]...")
        for client in clients:
            client.send(None)
        sleep(self._timeout)

        clients = ep_select(clients)[0]
        self._logger.debug(f"Receiving local models from available requested clients [{len(clients)}]...")
        client_models = [model for model
                         in cast(list[list[np.ndarray]], receive_latest_ep_objs(clients, list).values())
                         if model is not None]

        if len(client_models) < min(1, int(num_clients * self._min_clients)):
            self._logger.info(f"Insufficient number of client models for aggregation received [{len(client_models)}]!")
            return
        self._logger.debug(f"Aggregating client models [{len(client_models)}] into global model...")
        global_model = self._m_aggr.aggregate(client_models)

        clients = self._aggr_serv.poll_connections()[1].values()
        self._logger.debug(f"Sending aggregated global model to all available clients [{len(clients)}]...")
        for client in clients:
            client.send(global_model)

    def _async_aggr(self):
        """Performs an asynchronous federated aggregation step, i.e. checking whether a sufficient number of clients
        sent their local models to the server, before aggregating their models, potentially also taking into account the
        global model of the previous step (depending on the model aggregator chosen). The newly created global model is
        sent back only to all clients that requested an aggregation step.
        """
        clients = self._aggr_serv.poll_connections()[0].values()
        if self._num_clients is not None and len(clients) < self._num_clients * self._min_clients:
            self._logger.info(f"Insufficient read-ready clients available for aggregation step [{len(clients)}]!")
            return

        self._logger.debug(f"Receiving local models from requesting clients [{len(clients)}]...")
        client_models = [model for model
                         in cast(list[list[np.ndarray]], receive_latest_ep_objs(clients, list).values())
                         if model is not None]

        if (len(client_models) == 0 or
                self._num_clients is not None and len(client_models) < self._num_clients * self._min_clients):
            self._logger.info(f"Insufficient number of client models for aggregation received [{len(client_models)}]!")
            return
        self._logger.debug(f"Aggregating client models [{len(client_models)}] into global model...")
        global_model = self._m_aggr.aggregate(client_models)

        clients = ep_select(clients)[1]
        self._logger.debug(f"Sending aggregated global model to available requesting clients [{len(clients)}]...")
        for client in clients:
            client.send(global_model)


class FederatedValueAggregator(FederatedOnlineAggregator):
    """Base class for generic value aggregation of messages which are sent continuously by the federated nodes to the
    aggregation server. These values are treated as some sort of series, each assigned to a federated node and its
    respective endpoint from which new values can be retrieved and stored in a sliding window queue.

    Since the content of messages sent by the federated nodes are shaped in various ways, depending on their type
    (potentially containing multiple values even), extensions of this class may be required, which should at least
    override the following methods:

        * process_node_msg(): Converts a singular received message from a federated node into a list of values to be
        added to the sliding window queue of that node. Per default this method assumes a singleton value.

    Note that the base class could be extended in various other ways as well (for example adding a dashboard), it is
    recommended to put such functionality into the setup() and cleanup() methods.
    """
    _aggr_values: dict[tuple[str, int], deque]
    _window_size: int

    def __init__(self, addr: tuple[str, int], name: str = "", timeout: int = 10, window_size: int = None):
        """Creates a new federated value aggregator.

        :param addr: Address of aggregation server for federated nodes to report to.
        :param name: Name of federated value aggregator for logging purposes.
        :param timeout: Timeout for waiting to receive message from federated nodes.
        :param window_size: Maximum number of latest received entries stored for each federated node.
        """
        super().__init__(addr=addr, name=name, timeout=timeout)

        self._window_size = window_size

    def setup(self):
        self._aggr_values = {}

    def cleanup(self):
        self._aggr_values = {}

    def create_fed_aggr(self):
        """Starts the loop to continuously poll the federated nodes for new values to receive and process, before adding
        them to the datastructure.
        """
        self._logger.info("Starting result aggregation loop...")
        while self._started:
            try:
                nodes = self._aggr_serv.poll_connections()[0].items()

                if len(nodes) == 0:
                    sleep(self._timeout)
                    continue

                for node, node_ep in nodes:
                    if node in self._aggr_values:
                        self._aggr_values[node] = deque(maxlen=self._window_size)
                    try:
                        while True:
                            new_values = self.process_node_msg(node, node_ep.receive(timeout=0))
                            self._aggr_values[node].extend(new_values)
                    except (RuntimeError, TimeoutError):
                        pass
            except RuntimeError:
                # stop() was called
                break
        self._logger.info("Result aggregation loop stopped.")

    def process_node_msg(self, node: tuple[str, int], msg) -> list:
        """Converts a singular received message from a federated node into a list of values to be
        added to the sliding window queue of that client. Per default this method assumes a singleton value.

        :param node: Node from whom the message was received.
        :param msg: Message to be processed.
        :return: List of values to be added to the sliding window of the respective node.
        """
        return [msg]


class FederatedPredictionAggregator(FederatedValueAggregator):
    """Aggregator for prediction values from a federated IDS. Since federated IDS nodes report their predictions in
    minibatches, each message received from a node contains a multitude of values, which must be fragmented first,
    before they can be stored.

    TODO @Seraphin
    """

    def __init__(self, addr: tuple[str, int], name: str = "", timeout: int = 10, window_size: int = None):
        """Creates a new federated prediction aggregator.

        :param addr: Address of aggregation server for federated nodes to report to.
        :param name: Name of federated value aggregator for logging purposes.
        :param timeout: Timeout for waiting to receive message from federated nodes.
        :param window_size: Maximum number of latest received entries stored for each federated node.
        """
        super().__init__(addr=addr, name=name, timeout=timeout, window_size=window_size)

    def setup(self):
        """

        """
        pass

    def cleanup(self):
        """

        """
        pass

    def process_node_msg(self, node: tuple[str, int], msg: tuple[np.ndarray, np.ndarray]) -> list:
        """Converts a received message from a federated node containing a minibatch of predictions into a list of tuples
        that each contain a datapoint and its respective prediction.

        :param node: Node from whom the message was received.
        :param msg: Minibatch arranged in x,y fashion to be disassembled into value pairs.
        :return: List of values (x,y) to be added to the sliding window of the respective node.
        """
        values = []
        x_data, y_pred = msg
        for i in range(len(x_data)):
            t = x_data[i], y_pred[i]
            self._logger.debug(f"Prediction received from {node}: {t}")
            values.append(t)
        return values


class FederatedEvaluationAggregator(FederatedValueAggregator):
    """TODO @Seraphin

    """

    def __init__(self, addr: tuple[str, int], name: str = "", timeout: int = 10, window_size: int = None):
        """Creates a new federated evaluator.

        :param addr: Address of aggregation server for federated nodes to report to.
        :param name: Name of federated value aggregator for logging purposes.
        :param timeout: Timeout for waiting to receive message from federated nodes.
        :param window_size: Maximum number of latest received entries stored for each federated node.
        """
        super().__init__(addr=addr, name=name, timeout=timeout, window_size=window_size)

    def setup(self):
        """

        """
        pass

    def cleanup(self):
        """

        """
        pass

    def process_node_msg(self, node: tuple[str, int], msg):
        """

        """
        pass
