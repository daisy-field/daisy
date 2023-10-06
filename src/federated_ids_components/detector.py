"""
    Class for Federated Client TODO

    Author: Fabian Hofmann
    Modified: 06.10.23
"""

import threading
from abc import ABC, abstractmethod
from time import sleep, time
import numpy as np
import tensorflow as tf
from typing import Callable

from src.communication import StreamEndpoint
from src.data_sources import DataSource
from src.federated_learning import FederatedModel
from src.federated_learning import ModelAggregator


class FederatedNode(ABC):
    """Abstract class for generic federated nodes, that learn cooperatively on generic streaming data using a generic
    model, while also running predictions on the samples of that data stream at every step. This class in its core wraps
    itself around various other classes of this framework to perform the following tasks:

        * DataSource: Source to draw data point samples from. Is done a number of times (batch size) before each step.
        * FederatedModel: Actual model to be fitted and run predictions alternatingly at each step at runtime.
        * StreamEndpoint: The generic servers for result reporting and evaluation, to be extended depending on topology.
            - Aggregator: Used to report prediction results to a centralized aggregation server (see aggregator.py).
            - Evaluator: Used to report prediction results to a centralized evaluation server (see evaluator.py).

    All of this is done in multithreaded manner, allowing to either implement synchronized or asynchronous federated
    approaches, with a centralized master node or any other topology for the nodes that learn together.

    TODO METHODS
    """
    _data_source: DataSource
    _batch_size: int
    _minibatch_inputs: list
    _minibatch_labels: list

    _model: FederatedModel
    _m_lock: threading.Lock

    _label_split: int
    _supervised: bool
    _metrics: list[tf.metrics]

    _eval_serv: StreamEndpoint
    _aggr_serv: StreamEndpoint

    _sync_mode: bool
    _update_interval_s: int
    _update_interval_t: int

    _s_since_update: int
    _t_last_update: float

    _learner_thread: threading.Thread
    _fed_thread: threading.Thread
    _started: bool

    def __init__(self, data_source: DataSource, batch_size: int, model: FederatedModel,
                 label_split: int = 2 ** 32, supervised: bool = False, metrics: list[tf.metrics] = None,
                 eval_server: StreamEndpoint = False, aggr_server: StreamEndpoint = False,
                 sync_mode: bool = True, update_interval_s: int = None, update_interval_t: int = None):
        """TODO commenting, checking, logging

        :param data_source:
        :param batch_size:
        :param model:
        :param label_split:
        :param supervised:
        :param metrics:
        :param eval_server:
        :param aggr_server:
        :param sync_mode:
        :param update_interval_s:
        :param update_interval_t:
        """
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

        self._eval_serv = eval_server
        self._aggr_serv = aggr_server

        self._sync_mode = sync_mode
        self._update_interval_s = update_interval_s
        self._update_interval_t = update_interval_t

        self._s_since_update = 0
        self._t_last_update = time()

        self._started = False

    def start(self):
        """TODO commenting, checking, logging

        """
        if self._started:
            raise RuntimeError(f"Federated node has already been started!")
        self._started = True
        _try_ops(
            self._data_source.open,
            self._eval_serv.start,
            self._aggr_serv.start
        )
        self.setup()

        self._learner_thread = threading.Thread(target=self.create_local_learner, daemon=True)
        self._learner_thread.start()
        if not self._sync_mode:
            self._fed_thread = threading.Thread(target=self.create_async_fed_learner, daemon=True)
            self._fed_thread.start()

    @abstractmethod
    def setup(self):
        """TODO commenting, checking, logging

        """
        raise NotImplementedError

    def stop(self):
        """TODO commenting, checking, logging

        """
        if not self._started:
            raise RuntimeError(f"Federated node has not been started!")
        self._started = False
        _try_ops(
            self._data_source.close,
            lambda: self._eval_serv.stop(shutdown=True),
            lambda: self._aggr_serv.stop(shutdown=True)
        )
        self.cleanup()

        self._learner_thread.join()
        if not self._sync_mode:
            self._fed_thread.join()

    @abstractmethod
    def cleanup(self):
        """TODO commenting, checking, logging

        """
        raise NotImplementedError

    def create_local_learner(self):
        """TODO commenting, checking, logging
        """
        for sample in self._data_source:  # FIXME SAMPLE SHOULD BE A TUPLE FOR SUPERVISED LEARNING?
            self._minibatch_inputs.append(sample[:self._label_split])
            self._minibatch_labels.append(sample[self._label_split:])

            if len(self._minibatch_inputs) > self._batch_size:
                x_data = np.array(self._minibatch_inputs)
                y_true = np.array(self._minibatch_labels)
                with self._m_lock:
                    try:
                        self._process_batch(x_data, y_true)
                    except RuntimeError:
                        # stop() was called
                        break
                self._minibatch_inputs = []
                self._minibatch_labels = []

                self._s_since_update += self._batch_size
                if self._sync_mode:
                    if (self._update_interval_s is not None and self._s_since_update > self._update_interval_s
                            or self._update_interval_t is not None
                            and time() - self._t_last_update > self._update_interval_t):
                        try:
                            self.sync_fed_update()
                        except RuntimeError:
                            # stop() was called
                            break
                        self._s_since_update = 0
                        self._t_last_update = time()

    def _process_batch(self, x_data, y_true):
        """TODO commenting, checking, logging

        :return:
        """
        y_pred = self._model.predict(x_data)
        if self._aggr_serv is not None:
            self._aggr_serv.send((x_data, y_pred))

        if len(self._metrics) > 0:
            eval_res = [metric(y_true, y_pred) for metric in self._metrics]
            if self._eval_serv is not None:
                self._eval_serv.send(eval_res)

        self._model.fit(x_data, y_true)

    @abstractmethod
    def sync_fed_update(self):
        """TODO commenting
        """
        raise NotImplementedError

    @abstractmethod
    def create_async_fed_learner(self):
        """TODO commenting
        """
        raise NotImplementedError

    def __del__(self):
        if self._started:
            self.stop()

class FederatedClient(FederatedNode):
    """TODO

    """
    _m_aggr_server: StreamEndpoint

    def __init__(self, is_sync: bool,
                 update_interval: int = -1):  # TODO THIS WONT WORK FOR SAMPLE VS TIME BASED SYNCHRONIZATION

        super().__init__()
        pass

class FederatedPeer(FederatedNode):
    """TODO

    """
    _peers: list[StreamEndpoint]  # TODO TBD by @Lotta
    _topology: object  # TODO TBD by @Lotta
    _aggr: ModelAggregator

    def __init__(self, ):
        super().__init__()
        pass

def _try_ops(*operations: Callable):
    """TODO commenting, checking, logging

    """
    for op in operations:
        try:
            op()
        except RuntimeError:
            pass
