#
#   Created: 2023
#   Author: Seraphin Zunzer
#   License: MIT License
# ---------------------------------------------------------------


# TODO: Transfer Learning Client
# TODO Get GAN size according to computational resources
# TODO Help other clients improving their models through transfer learning

import threading
from abc import ABC, abstractmethod
from time import sleep, time
import numpy as np
import tensorflow as tf
from typing import Callable

from src.communication import StreamEndpoint
from src.data_sources import DataSource
from src.transfer_learning.GANs import GenerativeAdversarialNetwork
from src.transfer_learning.resource_investigation import ResourceInvestigator


class TransferLearningNode(ABC):
    """Abstract class for generic transfer learning nodes, that learn cooperatively on generic streaming data using a GAN.
        TODO Discriminator has to be transferred to improve the performance see "Transferring GANs: generating images from limited data" (Wang et al +. 2018)
    """
    _data_source: DataSource
    _batch_size: int
    _minibatch_inputs: list
    _minibatch_labels: list

    _model: GenerativeAdversarialNetwork
    _m_lock: threading.Lock

    _peerNode: StreamEndpoint


    def __init__(self, data_source: DataSource, batch_size: int, model: GenerativeAdversarialNetwork,
                 label_split: int = 2 ** 32, supervised: bool = False, metrics: list[tf.metrics] = None,
                 peerNode: StreamEndpoint = False,
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

        self._peerNode = peerNode

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
            self._peerNode.start,
        )
        self.setup()

        self._learner_thread = threading.Thread(target=self.create_local_learner, daemon=True)
        self._learner_thread.start()
        if not self._sync_mode:
            self._fed_thread = threading.Thread(target=self.create_async_fed_learner, daemon=True)
            self._fed_thread.start()


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


    def _update_peer(self):
        #TODO Transmit part of the GAN to support learning process
        return NotImplementedError
    def _retrieve_knowledge(self):
        # TODO Receive part of the GAN and start learning
        return NotImplementedError

    def _scale_model(self):
        # TODO Scale GAN size according to available resources
        return NotImplementedError



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


    def __del__(self):
        if self._started:
            self.stop()

    def _try_ops(*operations: Callable):
        """TODO commenting, checking, logging

        """
        for op in operations:
            try:
                op()
            except RuntimeError:
                pass


class TransferLearningClient(TransferLearningNode):
    """TODO

    """
    _m_aggr_server: StreamEndpoint

    def __init__(self, is_sync: bool,
                 update_interval: int = -1):  # TODO THIS WONT WORK FOR SAMPLE VS TIME BASED SYNCHRONIZATION

        super().__init__()
        pass

