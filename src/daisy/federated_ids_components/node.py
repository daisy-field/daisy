# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""A collection of various types of federated worker nodes, implementing the same
interface for each federated node type. Each of them is able to learn cooperatively
on generic streaming data using a generic model, while also running predictions on
the samples of that data stream at every step.

Author: Fabian Hofmann
Modified: 04.04.24
"""
# TODO Future Work: Defining granularity of logging in inits
# TODO Future Work: Args for client-side ports in init

import logging
import sys
import threading
from abc import ABC, abstractmethod
from queue import Empty
from time import sleep, time
from typing import Callable, cast, Optional

import numpy as np
import tensorflow as tf

from daisy.chord import ChordDHTPeer
from daisy.communication import StreamEndpoint
from daisy.data_sources import (
    DataHandler,
)
from daisy.federated_learning import (
    FederatedModel,
    ModelAggregator,
)


# TODO Future Work: Defining granularity of logging in inits
# TODO Future Work: Args for client-side ports in init

from daisy.model_poisoning.model_poisoning import model_poisoning


class FederatedOnlineNode(ABC):
    """Abstract class for generic federated nodes, that learn cooperatively on
    generic streaming data using a generic model, while also running predictions on
    the samples of that data stream at every step. This class in its core wraps
    itself around various other classes of this framework to perform the following
    tasks:

        * DataHandler: Handler to draw data point samples from. Is done a number of
        times (batch size) before each step.

        * FederatedModel: Actual model to be fitted and run predictions alternatingly
        at each step at runtime.

        * StreamEndpoint: The generic aggregation servers for result reporting and
        evaluation, to be extended depending on topology (see aggregator.py).
            - Evaluator: Used to report evaluation results to a centralized server.
            - Aggregator: Used to report prediction results to a centralized server.

    All of this is done in multithreaded manner, allowing any implementation to be
    synchronously or asynchronously federated, with a centralized master node or any
    other topology for the nodes that learn together. To accomplish this,
    the following methods must be implemented:

        * setup(): Setup function for any state variables called during start,

        * cleanup(): Cleanup function for any stat variables called during stopping.

        * sync_fed_update(): Singular, synchronous federated update step.

        * create_async_fed_learner(): Continuous, asynchronous federated update loop.
    """

    _logger: logging.Logger

    _data_handler: DataHandler
    _batch_size: int
    _minibatch_inputs: list
    _minibatch_labels: list

    _model: FederatedModel
    _m_lock: threading.Lock

    _label_split: int
    _supervised: bool
    _metrics: list[tf.keras.metrics.Metric]

    _eval_serv: Optional[StreamEndpoint]
    _aggr_serv: Optional[StreamEndpoint]

    _sync_mode: bool
    _update_interval_s: int
    _update_interval_t: int
    _s_since_update: int
    _t_last_update: float
    _u_lock: threading.Lock

    _loc_learner: threading.Thread
    _fed_updater: threading.Thread
    _started: bool
    _completed = threading.Event

    def __init__(
        self,
        data_handler: DataHandler,
        batch_size: int,
        model: FederatedModel,
        name: str = "",
        label_split: int = 2**32,
        supervised: bool = False,
        metrics: list[tf.keras.metrics.Metric] = None,
        eval_server: tuple[str, int] = None,
        aggr_server: tuple[str, int] = None,
        sync_mode: bool = True,
        update_interval_s: int = None,
        update_interval_t: int = None,
    ):
        """Creates a new federated online node. Note that by default, the node never
        performs a federated update step; one of the intervals has to be set for that.

        :param data_handler: Data handler of data stream to draw data points from.
        :param batch_size: Minibatch size for each prediction-fitting step.
        :param model: Actual model to be fitted and run predictions on in online manner.
        :param name: Name of federated online node for logging purposes.
        :param label_split: Split index within data point vector between input and
        true label(s). Default is no labels.
        :param supervised: Learning mode for model (supervised/unsupervised). Default
        is unsupervised.
        :param metrics: Evaluation metrics to update at each step/minibatch. Default
        has no evaluation at all.
        :param eval_server: Address of evaluation server (see aggregator.py).
        :param aggr_server: Address of aggregation server (see aggregator.py).
        :param sync_mode: Federated updating mode for node (sync/async). Default is
        synchronized.
        :param update_interval_s: Federated updating interval, defined by samples;
        every X samples, do a sync update.
        :param update_interval_t: Federated updating interval, defined by time; every
        X seconds, do a sync update.
        """
        self._logger = logging.getLogger(name)
        self._logger.info("Initializing federated online node...")

        self._data_handler = data_handler
        self._batch_size = batch_size
        self._minibatch_inputs = []
        self._minibatch_labels = []

        self._model = model
        self._m_lock = threading.Lock()

        if label_split == 2**32 and (
            supervised or metrics is None or len(metrics) == 0
        ):
            raise ValueError("Supervised and/or evaluation mode requires labels!")
        self._label_split = label_split
        self._supervised = supervised
        self._metrics = metrics

        self._eval_serv = None
        if eval_server is not None:
            self._eval_serv = StreamEndpoint(
                name="EvalServer",
                remote_addr=eval_server,
                acceptor=False,
                multithreading=True,
            )
        self._aggr_serv = None
        if aggr_server is not None:
            self._aggr_serv = StreamEndpoint(
                name="AggrServer",
                remote_addr=aggr_server,
                acceptor=False,
                multithreading=True,
            )

        self._sync_mode = sync_mode
        self._update_interval_s = update_interval_s
        self._update_interval_t = update_interval_t
        self._s_since_update = 0
        self._t_last_update = time()
        self._u_lock = threading.Lock()

        self._started = False
        self._completed = threading.Event()
        self._logger.info("Federated online node initialized.")

    def start(self, blocking: bool = False):
        """Starts the federated online node, along with any underlying endpoints,
        data handler, and any other object by an extension of this class (see setup(
        )). Non-blocking.

        :param blocking: Whether the node should block until all data points have
        been processed.
        :return: Event object to check completion state of federated node, i.e. whether
        it has processed every data point and may be closed.
        :raises RuntimeError: If federated online node has already been started.
        """
        self._logger.info("Starting federated online node...")
        if self._started:
            raise RuntimeError("Federated online node has already been started!")
        self._started = True
        _try_ops(
            lambda: self._data_handler.open(),
            lambda: self._eval_serv.start(),
            lambda: self._aggr_serv.start(),
            logger=self._logger,
        )
        self._logger.info("Performing further setup...")
        self.setup()

        self._loc_learner = threading.Thread(
            target=self._create_loc_learner, daemon=True
        )
        self._loc_learner.start()
        if not self._sync_mode:
            self._logger.info("Async learning detected, starting fed learner thread...")
            self._fed_updater = threading.Thread(
                target=self.create_async_fed_learner, daemon=True
            )
            self._fed_updater.start()
        self._logger.info("Federated online node started.")

        if blocking:
            self._completed.wait()
            self._logger.info("Node has processed all data points and may be closed.")
        return self._completed

    @abstractmethod
    def setup(self):
        """Setup function that must be implemented, called during start(); sets any
        new internal state variables and objects up used during the federated
        updating process, both for synchronous and asynchronous learning.

        Note that any such instance attribute should be initialized within an
        extension of the __init_() method.
        """
        raise NotImplementedError

    def stop(self):
        """Stops the federated online node, along with any underlying endpoints,
        data handlers, and any other object by an extension of this class (see
        cleanup()).

        :raises RuntimeError: If federated online node has not been started.
        """
        self._logger.info("Stopping federated online node...")
        if not self._started:
            raise RuntimeError("Federated online node has not been started!")
        self._started = False
        _try_ops(
            lambda: self._data_handler.close(),
            lambda: self._eval_serv.stop(shutdown=True),
            lambda: self._aggr_serv.stop(shutdown=True),
            logger=self._logger,
        )
        self._logger.info("Performing further cleanup...")
        self.cleanup()

        self._loc_learner.join()
        if not self._sync_mode:
            self._logger.info(
                "Async learning detected, waiting for fed learner thread to stop..."
            )
            self._fed_updater.join()
        self._logger.info("Federated online node stopped.")

    @abstractmethod
    def cleanup(self):
        """Cleanup function that must be implemented, called during stop(); resets
        any new internal state variables and objects up used during the federated
        updating process, both for synchronous and asynchronous learning.
        """
        raise NotImplementedError

    def _create_loc_learner(self):
        """Starts the loop to retrieve samples from the data handler, arranging them
        into minibatches and running predictions and fittings on them and the
        federated model. If set, also initiates synchronous federated update steps if
        sample/time intervals are satisfied.
        """
        self._logger.info("AsyncLearner: Starting...")
        try:
            for sample in self._data_handler:
                self._logger.debug(
                    "AsyncLearner: Appending sample to current minibatch..."
                )
                self._minibatch_inputs.append(sample[: self._label_split])
                self._minibatch_labels.append(sample[self._label_split :])

                if len(self._minibatch_inputs) > self._batch_size:
                    self._logger.debug("AsyncLearner: Processing full minibatch...")
                    with self._m_lock:
                        self._process_batch()
                    with self._u_lock:
                        self._s_since_update += self._batch_size

                if self._sync_mode:
                    self.sync_fed_update_check()
        except RuntimeError:
            # stop() was called
            pass
        self._logger.info("AsyncLearner: Data source exhausted, or node closed.")
        self._completed.set()

    def _process_batch(self):
        """Processes the current batch for both running a prediction and fitting the
        federated model around it. Also sends results to both the aggregation and the
        evaluation server, if available and provided in the beginning,
        before flushing the minibatch window.
        """
        self._logger.debug("AsyncLearner: Arranging full minibatch for processing...")
        x_data, y_true = (
            np.array(self._minibatch_inputs),
            np.array(self._minibatch_labels),
        )

        self._logger.debug("AsyncLearner: Processing minibatch...")
        y_pred = self._model.predict(x_data)
        self._logger.debug(
            f"AsyncLearner: Prediction results for minibatch: {(x_data, y_pred)}"
        )
        if self._aggr_serv is not None:
            self._aggr_serv.send((x_data, y_pred))
        if len(self._metrics) > 0:
            try:
                eval_res = {
                    metric.name: metric(y_true, y_pred) for metric in self._metrics
                }
            except ValueError as e:
                self._logger.error(e)
                self._logger.error(
                    "There occurred an error when comparing predictions with true labels. Check if the labels of the used dataset are correctly processed and if the label_split variable was set to the right value when configuring the client"
                )
                sys.exit(1)

            self._logger.error(
                f"AsyncLearner: Evaluation results for minibatch: {eval_res}"
            )
            self._logger.debug(
                f"AsyncLearner: Evaluation results for minibatch: {eval_res}"
            )
            if self._eval_serv is not None:
                self._eval_serv.send(
                    {
                        metric.name: {x: y.numpy() for x, y in metric.result().items()}
                        for metric in self._metrics
                    }
                )

        self._model.fit(x_data, y_true)
        self._logger.debug("AsyncLearner: Minibatch processed, cleaning window ...")
        self._minibatch_inputs, self._minibatch_labels = [], []

    def sync_fed_update_check(self):
        """Checks whether the conditions for a synchronous federated update step are
        met and performs it."""
        if (
            self._update_interval_s is not None
            and self._s_since_update > self._update_interval_s
            or self._update_interval_t is not None
            and time() - self._t_last_update > self._update_interval_t
        ):
            self._logger.debug(
                "AsyncLearner: Initiating synchronous federated update step..."
            )
            self.fed_update()
            self._s_since_update = 0
            self._t_last_update = time()

    @abstractmethod
    def fed_update(self):
        """Singular, synchronous federated update step for the underlying model of
        the federated online node. Encapsulates all that is necessary,
        from communication to other nodes, to transferring of one's own model (if
        necessary) to the model update itself.

        Note that any update of federated models while fitting or prediction is done
        will result in race conditions and unsafe states! It is therefore crucial to
        use the _m_lock instance variable to synchronize access to the model. Using
        this lock object, one can also manage when and how the other thread is able
        to use the model during any update step (if updating is done in
        semi-synchronous manner).

        Note this could also be used for semi-async federated update steps, where the
        updating happens asynchronously, however the update step within a single
        federated node is synchronous. If this is desired, create_async_fed_learner()
        could use this method for the actual update procedure.
        """
        raise NotImplementedError

    @abstractmethod
    def create_async_fed_learner(self):
        """Continuous, asynchronous federated update loop, that runs concurrently to
        the thread of create_local_learner(), to update the underlying model of the
        federated online node. Must use the _started semaphore to exit the loop in
        case the node is stopped (see stop()).

        Note that any update of federated models while fitting or prediction is done
        will result in race conditions and unsafe states! It is therefore crucial to
        use the _m_lock instance variable to synchronize access to the model. Using
        this lock object, one can also manage when and how the other thread is able
        to use the model during any update step (if updating is done in
        semi-synchronous manner).

        For coordination/planning when to perform an update, any implementation can
        also use existing state variables also used in the synchronous
        create_local_learner(), see: _update_interval_s, _update_interval_t,
        _s_since_update, _t_last_update, for sample- or time-based updating periods.
        Access to these variables must also be synchronized, using the _u_lock
        instance variable.
        """
        raise NotImplementedError

    def __del__(self):
        if self._started:
            self.stop()


class FederatedOnlineClient(FederatedOnlineNode):
    """Centralized federated learning is the simplest federated approach, since any
    client in the topology, while learning by itself, always reports to the same
    centralized server that aggregate the models in their stead.

    This implementation follows the FedAvg approach, i.e. the client reports its
    model's parameters to the server either in fixed intervals, either time-based or
    sample-based (like the implemented abstract class), or when called upon (sampled
    FedAvg), before receiving the new global model that replaces the local one.
    However, it is not fixed how the model aggregation server decides how the
    different models get aggregated; whether it happens in synchronized fashion or
    for each reporting client individually (see aggregator.py)
    """

    _m_aggr_server: StreamEndpoint
    _timeout: int

    def __init__(
        self,
        data_handler: DataHandler,
        batch_size: int,
        model: FederatedModel,
        m_aggr_server: tuple[str, int],
        timeout: int = 10,
        name: str = "",
        label_split: int = 2**32,
        supervised: bool = False,
        metrics: list[tf.keras.metrics.Metric] = None,
        eval_server: tuple[str, int] = None,
        aggr_server: tuple[str, int] = None,
        sync_mode: bool = True,
        update_interval_s: int = None,
        update_interval_t: int = None,
        poisoning_mode: str = None,
    ):
        """Creates a new federated online client.

        :param data_handler: Data handler of data stream to draw data points from.
        :param batch_size: Minibatch size for each prediction-fitting step.
        :param model: Actual model to be fitted and run predictions on in online manner.
        :param m_aggr_server: Address of centralized model aggregation server
        (see aggregator.py).
        :param timeout: Timeout for waiting to receive global model updates from
        model aggregation server.
        :param name: Name of federated online node for logging purposes.
        :param label_split: Split index within data point vector between input and
        true label(s). Default is no labels.
        :param supervised: Learning mode for model (supervised/unsupervised). Default
        is unsupervised.
        :param metrics: Evaluation metrics to update at each step/minibatch. Default
        has no evaluation at all.
        :param eval_server: Address of evaluation server (see aggregator.py).
        :param aggr_server: Address of aggregation server (see aggregator.py).
        :param sync_mode: Federated updating mode for node (sync/async). Default is
        synchronized. If async, allows the aggregation server to trigger an update
        step if neither of the intervals are set.
        :param update_interval_s: Federated updating interval, defined by samples;
        every X samples, do an update step.
        :param update_interval_t: Federated updating interval, defined by time; every
        X seconds, do an update step.
        """
        super().__init__(
            data_handler=data_handler,
            batch_size=batch_size,
            model=model,
            name=name,
            label_split=label_split,
            supervised=supervised,
            metrics=metrics,
            eval_server=eval_server,
            aggr_server=aggr_server,
            sync_mode=sync_mode,
            update_interval_s=update_interval_s,
            update_interval_t=update_interval_t,
        )

        self._m_aggr_server = StreamEndpoint(
            name="MAggrServer",
            remote_addr=m_aggr_server,
            acceptor=False,
            multithreading=True,
        )
        self._timeout = timeout
        self._poisoningMode = poisoning_mode

    def setup(self):
        _try_ops(lambda: self._m_aggr_server.start(), logger=self._logger)

    def cleanup(self):
        _try_ops(lambda: self._m_aggr_server.stop(shutdown=True), logger=self._logger)

    def fed_update(self):
        """Sends the client's local model's parameters to the model aggregation
        server, before receiving the global model, and updating the local one with
        it. Note this indeed blocks the local learner thread from processing further
        even when async is enabled, since the model cannot be used during update
        periods.

        Note that even though the endpoint seemingly guarantees that messages are
        delivered and also in order, the remote aggregation server could also fail,
        discard messages due to an overflow on the application layer, etc.. This
        means for a truly synchronized federated update to take place, clients must
        allow federated update steps to be skipped, since otherwise, it would block
        online detection/learning.
        """
        with self._m_lock:
            current_params = self._model.get_parameters()
            self._logger.info("Model Parameters:")
            for i in current_params:
                if not (isinstance(i, int) or isinstance(i, float)):
                    self._logger.info(i.shape)
                else:
                    self._logger.info(i)

            poisoned_params = model_poisoning(
                current_params, self._poisoningMode, model=self._model
            )
            self._m_aggr_server.send(poisoned_params)

            self._logger.debug(
                "Receiving global model parameters from model aggregation server..."
            )
            try:
                m_aggr_msg = self._m_aggr_server.receive(timeout=self._timeout)
                if not isinstance(m_aggr_msg, list):
                    self._logger.warning(
                        "Received message from model aggregation server "
                        "does not contain parameters!"
                    )
                    return
            except TimeoutError:
                self._logger.warning(
                    "Timeout triggered. No message received "
                    "from model aggregation server!"
                )
                return

            if self._poisoningMode is None or self._poisoningMode == "inverse":
                self._logger.debug("Updating local model with global parameters...")
                new_params = cast(np.array, m_aggr_msg)
                self._model.set_parameters(new_params)

    def create_async_fed_learner(self):
        """Starts the loop to check whether the conditions for a synchronized
        federated updating step are met, before initiating the update. This is either
        done based on set sample/time intervals or when called upon by the
        aggregation server.
        """
        self._logger.info("AsyncUpdater: Starting...")
        while self._started:
            self._logger.debug(
                "AsyncUpdater: Performing federated update step checks..."
            )
            try:
                if self._update_interval_t is not None:
                    self._logger.debug(
                        "AsyncUpdater: Time-based federated updating initiated..."
                    )
                    self.fed_update()
                    sleep(self._update_interval_t)
                elif self._update_interval_s is not None:
                    with self._u_lock:
                        if self._s_since_update > self._update_interval_s:
                            self._logger.debug(
                                "AsyncUpdater: Sample-based federated "
                                "updating initiated..."
                            )
                            self.fed_update()
                            self._s_since_update = 0
                    sleep(1)
                else:
                    self._logger.debug(
                        "AsyncUpdater: Sampled federated updating initiated..."
                    )
                    self.sampled_fed_update()
            except RuntimeError:
                # stop() was called
                break
        self._logger.info("AsyncUpdater: Stopping...")

    def sampled_fed_update(self):
        """Performs a sampled federated update step, waiting for the model
        aggregation server to call upon the federated (client) node, before either
        updating the model with the parameters received directly or initiating the
        updating step.
        """
        self._logger.debug(
            "AsyncUpdater: Receiving message from model aggregation server..."
        )
        try:
            m_aggr_msg = self._m_aggr_server.receive(timeout=self._timeout)
        except TimeoutError:
            self._logger.warning(
                "AsyncUpdater: Timeout triggered. No message received "
                "from model aggregation server!"
            )
            return
        if isinstance(m_aggr_msg, list):
            with self._m_lock:
                self._logger.debug(
                    "AsyncUpdater: Updating local model "
                    "with received global parameters..."
                )
                new_params = cast(np.array, m_aggr_msg)
                self._model.set_parameters(new_params)
        else:
            self._logger.debug(
                "AsyncUpdater: Request for sampled federated update received..."
            )
            self.fed_update()


class FederatedOnlinePeer(FederatedOnlineNode):
    """TODO by @lotta
    TODO Tit for Tat parameter
    """

    _topology: ChordDHTPeer
    _topology_thread: threading.Thread
    _addr: tuple[str, int]
    _dht_join_addr: tuple[str, int]

    _m_aggr: ModelAggregator
    _num_peers: int
    _choked_peers: set

    model: FederatedModel

    _logger: logging.Logger

    def __init__(
        self,
        data_handler: DataHandler,
        batch_size: int,
        model: FederatedModel,
        m_aggr: ModelAggregator,
        port: int,
        name: str = "",
        label_split: int = 2**32,
        supervised: bool = False,
        metrics: list[tf.keras.metrics.Metric] = None,
        eval_server: tuple[str, int] = None,
        aggr_server: tuple[str, int] = None,
        update_interval_s: int = None,
        update_interval_t: int = None,
        dht_join_addr: tuple[str, int] = None,
        num_peers: int = 4,
    ):
        """Creates a new federated online peer.

        :param data_handler: Data handler of data stream to draw data points from.
        :param batch_size: Minibatch size for each prediction-fitting step.
        :param model: Actual model to be fitted and run predictions on in online manner.
        :param m_aggr: Model aggregator for local model aggregation.
        :param name: Name of federated online node for logging purposes.
        :param label_split: Split index within data point vector between input and
        true label(s). Default is no labels.
        :param supervised: Learning mode for model (supervised/unsupervised). Default
        is unsupervised.
        :param metrics: Evaluation metrics to update at each step/minibatch. Default
        has no evaluation at all.
        :param eval_server: Address of evaluation server (see aggregator.py).
        :param aggr_server: Address of aggregation server (see aggregator.py).
        :param sync_mode: Federated updating mode for node (sync/async). Default is
        synchronized.
        :param update_interval_s: Federated updating interval, defined by samples;
        every X samples, do a sync update.
        :param update_interval_t: Federated updating interval, defined by time; every
        X seconds, do a sync update.
        """
        super().__init__(
            data_handler=data_handler,
            batch_size=batch_size,
            model=model,
            name=name,
            label_split=label_split,
            supervised=supervised,
            metrics=metrics,
            eval_server=eval_server,
            aggr_server=aggr_server,
            sync_mode=False,
            update_interval_s=update_interval_s,
            update_interval_t=update_interval_t,
        )
        self._addr = ("127.0.0.1", port)
        self._dht_join_addr = dht_join_addr
        self._m_aggr = m_aggr
        self._topology = ChordDHTPeer(addr=self._addr, cluster_size=num_peers)
        self._num_peers = num_peers
        self._choked_peers = set()
        self._started = False

        self._logger = logging.getLogger("FED_NODE")
        self._logger.setLevel(logging.INFO)

    def setup(self):
        """ """
        # start dht
        self._topology_thread = threading.Thread(
            target=lambda: self._topology.start(self._dht_join_addr), daemon=True
        )
        self._topology_thread.start()
        return

    def cleanup(self):
        """TODO implement"""
        raise NotImplementedError

    def fed_update(self, models: list[list[np.ndarray]] = None):
        """ """
        if models is not None and len(models) > 0:
            self._logger.info(
                f"Calculating federated Update with {len(models)} models."
            )
            self._model.set_parameters(self._m_aggr.aggregate(models))
            self._logger.info("Model metrics: ")

    def create_async_fed_learner(self):
        """ """
        # TODO sample based/ time based
        # TODO favoritenliste mit peers die schonal geantwortet haben
        #  wann/wie oft wird optimistic unchoking durchgeführt,
        #  wann/wie oft tit for tat -> Paper sagt alle 10 sek. default
        start = time()
        tft_timer = start
        auto_modelsharing_timer = start
        full_unchoke_timer = start
        models = []
        while self._started:
            if time() - full_unchoke_timer >= 60:
                self._choked_peers.clear()
            if time() - auto_modelsharing_timer >= 30:
                self._send_model_and_choke_receiving_peer()
                auto_modelsharing_timer = time()
            if time() - tft_timer >= 10:
                models = self._tit_for_tat()
                tft_timer = time()
            if models and len(models) >= 4:
                with self._m_lock:
                    models.append(self._model.get_parameters())
                    self.fed_update(models)
                models.clear()
            else:
                sleep(1)

    def _send_model_and_choke_receiving_peer(self):
        # best effort solution, take what is there
        peers = self._get_peers_for_cluster()
        for peer in peers:
            if peer in self._choked_peers:
                continue
            with self._m_lock:
                self._topology.fed_models_outgoing.put(
                    (peer, self._model.get_parameters())
                )
                self._choked_peers.add(peer)

    def _get_peers_for_cluster(self):
        peers = set()
        while len(peers) < self._num_peers:
            try:
                peers.add(self._topology.fed_peers.get_nowait())
            except Empty:
                self._logger.info(
                    f"Sampled {len(self._choked_peers)} Peers from Network."
                )
                return peers

    def _tit_for_tat(self) -> list[list[np.ndarray]]:
        fed_peers_and_models = self._get_peer_models_from_topology_queue()
        fed_models = []
        for fed_peer in fed_peers_and_models:
            fed_model = fed_peers_and_models[fed_peer]
            if fed_peer in self._choked_peers:
                self._choked_peers.remove(fed_peer)
            else:
                self._topology.fed_models_outgoing.put((fed_peer, fed_model))
            fed_models.append(fed_model)
        return fed_models

    def _get_peer_models_from_topology_queue(self):
        fed_peers_and_models = {}
        while len(fed_peers_and_models) < self._num_peers:
            try:
                fed_peer, fed_model = self._topology.fed_models_incoming.get_nowait()
                fed_peers_and_models[fed_peer] = fed_model
            except Empty:
                self._logger.info(
                    f"Retrieved {len(fed_peers_and_models)} Models from network."
                )
                return fed_peers_and_models


def _try_ops(*operations: Callable, logger: logging.Logger):
    """Takes a number of callable objects and executes them sequentially, each time
    logging any arising attribute and runtime errors. Caution is advised when using
    this function, especially if the potential erroneous behavior is not expected!

    :param operations: One or multiple callables that are to be executed.
    :param logger: Logger to log any error in full detail.
    """
    for op in operations:
        try:
            op()
        except (AttributeError, RuntimeError) as e:
            logger.warning(
                f"{e.__class__.__name__}({e}) while trying to execute {op}: {e}"
            )
