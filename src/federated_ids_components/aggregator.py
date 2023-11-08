"""
    Class for Federated Aggregator TODO

    Author: Fabian Hofmann
    Modified: 27.10.23
"""
import logging
import threading
from abc import ABC, abstractmethod
from time import time

from src.communication import EndpointServer
from src.federated_learning import FederatedModel, ModelAggregator


class FederatedOnlineAggregator(ABC):
    """TODO

    """
    _logger: logging.Logger

    _aggr_serv: EndpointServer

    _fed_aggr: threading.Thread
    _started: bool

    def __init__(self, addr: tuple[str, int], name: str = ""):
        """TODO

        :param addr: TODO
        :param name: Name of federated online node for logging purposes.
        """
        self._logger = logging.getLogger(name)
        self._logger.info("Initializing federated online node...")

        self._aggr_serv = EndpointServer(name="Server", addr=addr, c_timeout=60, multithreading=True)

        self._started = False
        self._logger.info("Federated online node initialized.")

    def start(self):
        """TODO

        Starts the federated online node, along with any underlying endpoints, data sources, and any other object by
        an extension of this class (see setup()). Non-blocking.

        :raises RuntimeError: If federated online node has already been started.
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
        """TODO

        Setup function that must be implemented, called during start(); sets any new internal state variables and
        objects up used during the federated updating process, both for synchronous and asynchronous learning.

        Note that any such instance attribute should be initialized within an extension of the __init_() method.
        """
        raise NotImplementedError

    def stop(self):
        """TODO

        Stops the federated online node, along with any underlying endpoints, data sources, and any other object by
        an extension of this class (see cleanup()).

        :raises RuntimeError: If federated online node has not been started.
        """
        self._logger.info("Stopping federated online aggregator...")
        if not self._started:
            raise RuntimeError(f"Federated online aggregator has not been started!")
        self._started = False
        try:
            self._aggr_serv.stop(),
        except RuntimeError:
            pass
        self._logger.info("Performing further cleanup...")
        self.cleanup()

        self._fed_aggr.join()
        self._logger.info("Federated online aggregator stopped.")

    @abstractmethod
    def cleanup(self):
        """TODO

        Cleanup function that must be implemented, called during stop(); resets any new internal state variables and
        objects up used during the federated updating process, both for synchronous and asynchronous learning.
        """
        raise NotImplementedError

    @abstractmethod
    def create_fed_aggr(self):
        """TODO

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
        raise NotImplementedError

    def __del__(self):
        if self._started:
            self.stop()


class FederatedModelAggregator(FederatedOnlineAggregator):
    """TODO
    """
    _model: FederatedModel
    _m_aggr: ModelAggregator

    _timeout: int

    _update_interval_t: int
    _t_last_update: float

    _sampled_update: bool

    def __init__(self, model: FederatedModel, m_aggr: ModelAggregator, addr: tuple[str, int], name: str = "",
                 timeout: int = 10, sampled_update: bool = False, update_interval_t: int = None):
        """TODO

        :param model: Actual model to be fitted and run predictions alternatingly in online manner.
        :param m_aggr: TODO
        :param addr: TODO
        :param name: Name of federated online node for logging purposes.
        :param timeout: Timeout for waiting to receive global model updates from model aggregation server.
        :param sampled_update: If async, allows the aggregation server to trigger a sync update (instead of intervals).
        :param update_interval_t: Federated updating interval, defined by time; every X seconds, do a sync update.
        """
        super().__init__(addr=addr, name=name)

        self._model = model
        self._m_aggr = m_aggr


        self._timeout = timeout
        self._update_interval_t = update_interval_t
        self._t_last_update = time()

    @abstractmethod
    def setup(self):
        """Setup function that must be implemented, called during start(); sets any new internal state variables and
        objects up used during the federated updating process, both for synchronous and asynchronous learning.

        Note that any such instance attribute should be initialized within an extension of the __init_() method.
        """
        raise NotImplementedError

    @abstractmethod
    def cleanup(self):
        """TODO

        Cleanup function that must be implemented, called during stop(); resets any new internal state variables and
        objects up used during the federated updating process, both for synchronous and asynchronous learning.
        """
        raise NotImplementedError


class FederatedResultAggregator:
    """TODO

    """
    pass
