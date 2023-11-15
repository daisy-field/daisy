"""
    Class for Federated Aggregator TODO

    Author: Fabian Hofmann
    Modified: 27.10.23
"""
import logging
import threading
from abc import ABC, abstractmethod
from random import sample
from time import sleep
from typing import cast, Iterable, Sequence

from src.communication import EndpointServer, StreamEndpoint, ep_select
from src.federated_learning import ModelAggregator


class FederatedOnlineAggregator(ABC):
    """Abstract class for generic federated online aggregators, that receive and aggregates data from federated nodes at
    runtime in a client-server-based exchange, continuously. To accomplish this, the abstract class in its core is
    merely a wrapper around the EndpointServer class to process new incoming clients, leaving the actual functionality
    to its implementations, which run in multithreaded manner.

    To realize this, the following methods must be implemented:

        * setup(): Setup function for any state variables called during the starting of the starting of the aggregator.

        * cleanup(): Cleanup function for any stat variables called during the stopping of the aggregator.

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
        """Continuous, asynchronous federated aggregation loop, that must run over the entire life-cycle.
        """
        raise NotImplementedError

    def __del__(self):
        if self._started:
            self.stop()


class FederatedModelAggregator(FederatedOnlineAggregator):
    """TODO
    """
    _m_aggr: ModelAggregator

    _update_interval_t: int
    _num_clients: int
    _min_clients: float

    _online_update: bool

    def __init__(self, m_aggr: ModelAggregator, addr: tuple[str, int], name: str = "",
                 timeout: int = 10, online_update: bool = True,
                 update_interval_t: int = None, num_clients: int = None, min_clients: float = 1.0):
        """Creates a new federated model aggregator.

        TODO CHECKING, LOGGING, COMMENTS

        :param m_aggr: Actual aggregator to aggregate models with.
        :param addr: Address of aggregation server for clients to connect to.
        :param name: Name of federated online node for logging purposes.
        :param timeout: Timeout for waiting to receive global model updates from model aggregation server.
        :param online_update: If async, allows the aggregation server to trigger a sync update (instead of intervals).
        :param update_interval_t: Federated updating interval, defined by time; every X seconds, do a sync update.
        :param num_clients:
        :param min_clients:
        """
        super().__init__(addr=addr, name=name, timeout=timeout)

        self._m_aggr = m_aggr

        self._update_interval_t = update_interval_t
        self._num_clients = num_clients
        self._min_clients = min_clients

        self._online_update = online_update

    def setup(self):
        pass

    def cleanup(self):
        pass

    def create_fed_aggr(self):
        """TODO CHECKING, LOGGING, COMMENTS

        Continuous, asynchronous federated update loop, that runs concurrently to the thread of
        create_local_learner(), to update the underlying model of the federated online node.

        Note that any update of federated models while fitting or prediction is done will result in race conditions and
        unsafe states! It is therefore crucial to use the _m_lock instance variable to synchronize access to the model.
        Using this lock object, one can also manage when and how the other thread is able to use the model during any
        update step (if updating is done in semi-synchronous manner).

        For coordination/planning when to perform an update, any implementation can also use existing state variables
        also used in the synchronous create_local_learner(), see: _update_interval_s, _update_interval_t,
        _s_since_update, _t_last_update, for sample- or time-based updating periods. Access to these variables must also
        be synchronized, using the _u_lock instance variable.
        """
        self._logger.info("Starting...")
        while self._started:
            self._logger.debug("AsyncUpdater: Performing federated update step checks...")
            try:
                if self._update_interval_t is not None:  # Sync Update
                    self._logger.debug("AsyncUpdater: Time-based federated updating initiated...")
                    self._sync_aggr()
                    sleep(self._update_interval_t)

                elif self._online_update:  # FIXME
                    self._logger.debug("AsyncUpdater: Sampled federated updating initiated...")
                    self._async_aggr()

                else:  # FIXME
                    # Federated updating disabled
                    break
            except RuntimeError:
                # stop() was called
                break
        self._logger.info("Stopping...")

    def _sync_aggr(self):
        """TODO CHECKING, LOGGING, COMMENTS
        """
        clients = self._aggr_serv.poll_connections()[1].values()
        if self._num_clients is not None:
            if len(clients) < self._num_clients:
                return
            clients = sample(cast(Sequence, clients), self._num_clients)
        num_clients = len(clients)

        for client in clients:
            client.send(None)
        sleep(self._timeout)

        clients = ep_select(clients)[0]
        client_models = self.get_client_models(clients)

        if len(client_models) < self._min_clients * num_clients:
            return
        global_model = self._m_aggr.aggregate(client_models)
        _, w_ready = self._aggr_serv.poll_connections()
        clients = w_ready.values()
        for client in clients:
            client.send(global_model)

    def _async_aggr(self):
        """TODO CHECKING, LOGGING, COMMENTS
        """
        clients = self._aggr_serv.poll_connections()[0].values()
        if self._num_clients is not None and len(clients) < self._num_clients * self._min_clients:
            return

        client_models = self.get_client_models(clients)
        if (len(client_models) == 0 or
                self._num_clients is not None and len(client_models) < self._min_clients * self._num_clients):
            return
        global_model = self._m_aggr.aggregate(client_models)
        clients = ep_select(clients)[1]
        for client in clients:
            client.send(global_model)

    def get_client_models(self, clients: Iterable[StreamEndpoint]):
        """TODO CHECKING, LOGGING, COMMENTS
        """
        client_models = []
        for client in clients:
            client_model = None
            try:
                while True:
                    client_msg = client.receive(timeout=0)
                    if isinstance(client_msg, list):
                        client_model = client_msg
                    else:
                        pass  # FIXME logging
            except (RuntimeError, TimeoutError):
                pass # FIXME logging
            if client_model is not None:
                client_models.append(client_model)
        return client_models


class FederatedResultAggregator(FederatedOnlineAggregator):
    """TODO

    """

    def setup(self):
        pass

    def cleanup(self):
        pass

    def create_fed_aggr(self):
        pass
