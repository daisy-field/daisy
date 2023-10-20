"""
    Class for Federated Client TODO

    Author: Fabian Hofmann
    Modified: 06.10.23
"""

import logging
import threading
from abc import ABC, abstractmethod
from time import sleep, time
from typing import Callable, cast, Optional

import numpy as np
import tensorflow as tf

from src.communication import StreamEndpoint
from src.data_sources import DataSource
from src.federated_learning import FederatedModel
from src.federated_learning import ModelAggregator


class FederatedOnlineNode(ABC):
    """Abstract class for generic federated nodes, that learn cooperatively on generic streaming data using a generic
    model, while also running predictions on the samples of that data stream at every step. This class in its core wraps
    itself around various other classes of this framework to perform the following tasks:

        * DataSource: Source to draw data point samples from. Is done a number of times (batch size) before each step.

        * FederatedModel: Actual model to be fitted and run predictions alternatingly at each step at runtime.

        * StreamEndpoint: The generic servers for result reporting and evaluation, to be extended depending on topology.
            - Evaluator: Used to report prediction results to a centralized evaluation server (see evaluator.py).
            - Aggregator: Used to report prediction results to a centralized aggregation server (see aggregator.py).

    All of this is done in multithreaded manner, allowing any implementation to be synchronously or asynchronously
    federated, with a centralized master node or any other topology for the nodes that learn together. To accomplish
    this, the following methods must be implemented:

        * setup(): Setup function for any state variables called during the starting of the starting of the node.

        * cleanup(): Cleanup function for any stat variables called during the starting of the starting of the node.

        * sync_fed_update(): Singular, synchronous federated update step.

        * create_async_fed_learner(): Continuous, asynchronous federated update loop.
    """
    _logger: logging.Logger

    _data_source: DataSource
    _batch_size: int
    _minibatch_inputs: list
    _minibatch_labels: list

    _model: FederatedModel
    _m_lock: threading.Lock

    _label_split: int
    _supervised: bool
    _metrics: list[tf.metrics]

    _eval_serv: Optional[StreamEndpoint]
    _aggr_serv: Optional[StreamEndpoint]

    _sync_mode: bool
    _update_interval_s: int
    _update_interval_t: int
    _u_lock: threading.Lock

    _s_since_update: int
    _t_last_update: float

    _learner_t: threading.Thread
    _fed_updater: threading.Thread
    _started: bool

    def __init__(self, data_source: DataSource, batch_size: int, model: FederatedModel,
                 name: str = "",
                 label_split: int = 2 ** 32, supervised: bool = False, metrics: list[tf.metrics] = None,
                 eval_server: tuple[str, int] = None, aggr_server: tuple[str, int] = None,
                 sync_mode: bool = True, update_interval_s: int = None, update_interval_t: int = None):
        """Creates a new federated online node.

        :param data_source: Data source of data stream to draw data points (in order) from.
        :param batch_size: Minibatch size for each prediction-fitting step.
        :param model: Actual model to be fitted and run predictions alternatingly in online manner.
        :param name: Name of federated online node for logging purposes.
        :param label_split: Split index within data point vector between input and true label(s). Default is no labels.
        :param supervised: Learning mode for model (supervised/unsupervised). Default is unsupervised.
        :param metrics: Evaluation metrics to update at each step/minibatch. Default has no evaluation at all.
        :param eval_server: Address of centralized evaluation server (see evaluator.py).
        :param aggr_server: Address of centralized aggregation server (see aggregator.py).
        :param sync_mode: Federated updating mode for node (sync/async). Default is synchronized.
        :param update_interval_s: Federated updating interval, defined by samples; every X samples, do a sync update.
        :param update_interval_t: Federated updating interval, defined by time; every X seconds, do a sync update.
        """
        self._logger = logging.getLogger(name)
        self._logger.info("Initializing federated online node...")

        self._data_source = data_source
        self._batch_size = batch_size
        self._minibatch_inputs = []
        self._minibatch_labels = []

        self._model = model
        self._m_lock = threading.Lock()

        if label_split == 2 ** 32 and (supervised or len(metrics) == 0):
            raise ValueError("Supervised and/or evaluation mode requires labels!")
        self._label_split = label_split
        self._supervised = supervised
        self._metrics = metrics

        self._eval_serv = None
        if eval_server is not None:
            self._eval_serv = StreamEndpoint(name="EvalServer", addr=("127.0.0.1", 20000), remote_addr=eval_server,
                                             acceptor=False, multithreading=True)
        self._aggr_serv = None
        if aggr_server is not None:
            self._aggr_serv = StreamEndpoint(name="AggrServer", addr=("127.0.0.1", 20001), remote_addr=aggr_server,
                                             acceptor=False, multithreading=True)

        self._sync_mode = sync_mode
        self._update_interval_s = update_interval_s
        self._update_interval_t = update_interval_t
        self._u_lock = threading.Lock()

        self._s_since_update = 0
        self._t_last_update = time()

        self._started = False
        self._logger.info("Federated online node initialized.")

    def start(self):
        """Starts the federated online node, along with any underlying endpoints, data sources, and any other object by
        an extension of this class (see setup()). Non-blocking.

        :raises RuntimeError: If federated online node has already been started.
        """
        self._logger.info("Starting federated online node...")
        if self._started:
            raise RuntimeError(f"Federated online node has already been started!")
        self._started = True
        _try_ops(
            self._data_source.open,
            self._eval_serv.start,
            self._aggr_serv.start,
            logger=self._logger
        )
        self._logger.info("Performing further setup...")
        self.setup()

        self._learner_t = threading.Thread(target=self._create_local_learner, daemon=True)
        self._learner_t.start()
        if not self._sync_mode:
            self._logger.info("Async learning detected, starting fed learner thread...")
            self._fed_updater = threading.Thread(target=self.create_async_fed_learner, daemon=True)
            self._fed_updater.start()
        self._logger.info("Federated online node started.")

    @abstractmethod
    def setup(self):
        """Setup function that must be implemented, called during start(); sets any new internal state variables and
        objects up used during the federated updating process, both for synchronous and asynchronous learning.

        Note that any such instance attribute should be initialized within an extension of the __init_() method.
        """
        raise NotImplementedError

    def stop(self):
        """Stops the federated online node, along with any underlying endpoints, data sources, and any other object by
        an extension of this class (see cleanup()).

        :raises RuntimeError: If federated online node has not been started.
        """
        self._logger.info("Stopping federated online node...")
        if not self._started:
            raise RuntimeError(f"Federated online node has not been started!")
        self._started = False
        _try_ops(
            self._data_source.close,
            lambda: self._eval_serv.stop(shutdown=True),
            lambda: self._aggr_serv.stop(shutdown=True),
            logger=self._logger
        )
        self._logger.info("Performing further cleanup...")
        self.cleanup()

        self._learner_t.join()
        if not self._sync_mode:
            self._logger.info("Async learning detected, waiting for fed learner thread to stop...")
            self._fed_updater.join()
        self._logger.info("Federated online node stopped.")

    @abstractmethod
    def cleanup(self):
        """Cleanup function that must be implemented, called during stop(); resets any new internal state variables and
        objects up used during the federated updating process, both for synchronous and asynchronous learning.
        """
        raise NotImplementedError

    def _create_local_learner(self):
        """Starts the loop to retrieve samples from the data source, arranging them into minibatches and running
        predictions and fittings on them and the federated model. If set, also initiates synchronous federated update
        steps if sample/time intervals are satisfied.
        """
        self._logger.info(f"AsyncLearner: Starting...")
        for sample in self._data_source:
            self._logger.debug(f"AsyncLearner: Appending sample to current minibatch...")
            self._minibatch_inputs.append(sample[:self._label_split])
            self._minibatch_labels.append(sample[self._label_split:])

            if len(self._minibatch_inputs) > self._batch_size:
                self._logger.debug(f"AsyncLearner: Arranging full minibatch for processing...")
                x_data = np.array(self._minibatch_inputs)
                y_true = np.array(self._minibatch_labels)
                with self._m_lock:
                    try:
                        self._process_batch(x_data, y_true)
                    except RuntimeError:
                        # stop() was called
                        break
                self._logger.debug(f"AsyncLearner: Cleaning minibatch window...")
                self._minibatch_inputs = []
                self._minibatch_labels = []

                with self._u_lock:
                    self._s_since_update += self._batch_size
                if self._sync_mode:
                    if (self._update_interval_s is not None and self._s_since_update > self._update_interval_s
                            or self._update_interval_t is not None
                            and time() - self._t_last_update > self._update_interval_t):
                        self._logger.debug(f"AsyncLearner: Initiating synchronous federated update step...")
                        try:
                            self.sync_fed_update()
                        except RuntimeError:
                            # stop() was called
                            break
                        self._s_since_update = 0
                        self._t_last_update = time()
        self._logger.info(f"AsyncLearner: Stopping...")

    def _process_batch(self, x_data, y_true):
        """Processes a singular batch for both running a prediction and fitting the federated model around it. Also
        sends results to both the aggregation and the evaluation server, if available and provided in the beginning.

        :param x_data: Input data.
        :param y_true: Expected output, for supervised mode and/or evaluation purposes.
        """
        self._logger.debug(f"AsyncLearner: Processing minibatch...")
        y_pred = self._model.predict(x_data)
        self._logger.debug(f"AsyncLearner: Prediction results for minibatch: {y_pred}")
        if self._aggr_serv is not None:
            self._aggr_serv.send((x_data, y_pred))

        if len(self._metrics) > 0:
            eval_res = [metric(y_true, y_pred) for metric in self._metrics]
            self._logger.debug(f"AsyncLearner: Evaluation results for minibatch: {eval_res}")
            if self._eval_serv is not None:
                self._eval_serv.send(eval_res)

        self._model.fit(x_data, y_true)
        self._logger.debug(f"AsyncLearner: Minibatch processed...")

    @abstractmethod
    def sync_fed_update(self):
        """Singular, synchronous federated update step for the underlying model of the federated online node.
        Encapsulates all that is necessary, from communication to other nodes, to transferring of one's own model (if
        necessary) to the model update itself.
        """
        raise NotImplementedError

    @abstractmethod
    def create_async_fed_learner(self):
        """Continuous, asynchronous federated update loop, that runs concurrently to the thread of
        create_local_learner(), to update the underlying model of the federated online node.

        Note that any update of federated models while fitting or prediction is done will result in race conditions and
        unsafe states! It is therefore crucial to use the _m_lock instance variable to synchronize access to the model.
        Using this lock object, one can also manage when and how the other thread is able to use the model during any
        update step (if updating is done in semi-synchronous manner)

        For coordination/planning when to perform an update, any implementation can also use existing state variables
        also used in the synchronous create_local_learner(), see: _update_interval_s, _update_interval_t,
        _s_since_update, _t_last_update, for sample- or time-based updating periods. Access to these variables must also
        be synchronized, using the _u_lock instance variable.
        """
        raise NotImplementedError

    def __del__(self):
        if self._started:
            self.stop()


class FederatedOnlineClient(FederatedOnlineNode):
    """TODO
    MWE

    """
    _m_aggr_server: StreamEndpoint

    def __init__(self, data_source: DataSource, batch_size: int, model: FederatedModel, m_aggr_server: tuple[str, int],
                 name: str = "",
                 label_split: int = 2 ** 32, supervised: bool = False, metrics: list[tf.metrics] = None,
                 eval_server: tuple[str, int] = None, aggr_server: tuple[str, int] = None,
                 sync_mode: bool = True, update_interval_s: int = None, update_interval_t: int = None):
        """Creates a new federated online node. TODO

        :param data_source: Data source of data stream to draw data points (in order) from.
        :param batch_size: Minibatch size for each prediction-fitting step.
        :param model: Actual model to be fitted and run predictions alternatingly in online manner.
        :param m_aggr_server: TODO
        :param name: Name of federated online node for logging purposes.
        :param label_split: Split index within data point vector between input and true label(s). Default is no labels.
        :param supervised: Learning mode for model (supervised/unsupervised). Default is unsupervised.
        :param metrics: Evaluation metrics to update at each step/minibatch. Default has no evaluation at all.
        :param eval_server: Address of centralized evaluation server (see evaluator.py).
        :param aggr_server: Address of centralized aggregation server (see aggregator.py).
        :param sync_mode: Federated updating mode for node (sync/async). Default is synchronized.
        :param update_interval_s: Federated updating interval, defined by samples; every X samples, do a sync update.
        :param update_interval_t: Federated updating interval, defined by time; every X seconds, do a sync update.
        """
        super().__init__(data_source=data_source, batch_size=batch_size, model=model, name=name,
                         label_split=label_split, supervised=supervised, metrics=metrics,
                         eval_server=eval_server, aggr_server=aggr_server,
                         sync_mode=sync_mode, update_interval_s=update_interval_s, update_interval_t=update_interval_t)

        self._m_aggr_server = StreamEndpoint(name="MAggrServer", addr=("127.0.0.1", 20002), remote_addr=m_aggr_server,
                                             acceptor=False, multithreading=True)

    def setup(self):
        """TODO
        """
        _try_ops(
            lambda: self._m_aggr_server.start,
            logger=self._logger
        )

    def cleanup(self):
        """TODO
        """
        _try_ops(
            lambda: self._m_aggr_server.stop(shutdown=True),
            logger=self._logger
        )

    def sync_fed_update(self):
        """TODO
        """
        with self._m_lock:
            current_params = self._model.get_parameters()
            self._m_aggr_server.send(current_params)
            new_params = cast(np.array, self._m_aggr_server.receive())
            self._model.set_parameters(new_params)

    def create_async_fed_learner(self):
        """TODO


        Note not exact, only an approximate (due to switching) only model update is actually stopping the other thread
        from working with model, there can still be delays. minimum time when sync should happen!
        """
        while self._started:
            try:
                if self._update_interval_t is not None:
                    self._logger.debug(f"AsyncLearner: Initiating synchronous federated update step...")
                    self.sync_fed_update()
                    sleep(self._update_interval_t)
                elif self._update_interval_s is not None:
                    with self._u_lock:
                        if self._s_since_update > self._update_interval_s:
                            self._logger.debug(f"AsyncLearner: Initiating synchronous federated update step...")
                            self.sync_fed_update()
                            self._s_since_update = 0
                    sleep(1)
            except RuntimeError:
                # stop() was called
                break


class FederatedOnlinePeer(FederatedOnlineNode):
    """TODO

    """
    _peers: list[StreamEndpoint]  # TODO TBD by @Lotta
    _topology: object  # TODO TBD by @Lotta
    _aggr: ModelAggregator

    def __init__(self, ):
        """TODO
        """
        super().__init__()
        pass

    def setup(self):
        """TODO
        """
        pass

    def cleanup(self):
        """TODO
        """
        pass

    def sync_fed_update(self):
        """TODO
        """
        pass

    def create_async_fed_learner(self):
        """TODO
        """
        pass


def _try_ops(*operations: Callable, logger: logging.Logger):
    """Takes a number of callable objects and executes them sequentially, each time logging any arising attribute and
    runtime errors. Caution is advised when using this function, especially if the potential erroneous behavior is not
    expected!

    :param operations: One or multiple callables that are to be executed.
    :param logger: Logger to log any error in full detail.
    """
    for op in operations:
        try:
            op()
        except (AttributeError, RuntimeError) as e:
            logger.warning(f"{e.__class__.__name__}({e}) while trying to execute {op}: {e}")
