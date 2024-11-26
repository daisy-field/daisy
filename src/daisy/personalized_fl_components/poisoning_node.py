# Copyright (C) 2024 DAI-Labor and others
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

from time import sleep
from typing import cast

import numpy as np
import tensorflow as tf

from daisy.communication import StreamEndpoint
from daisy.data_sources import DataHandler
from daisy.federated_learning import FederatedModel

from daisy.federated_ids_components import FederatedOnlineNode
from daisy.federated_ids_components.node import _try_ops


class PoisoningClient(FederatedOnlineNode):
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
    _poisoningMode: str

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
        poisoningMode: str = "zeros",
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
        self._poisoningMode = poisoningMode

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
            poisoned_params = []
            if self._poisoningMode == "zeros":
                for layer in current_params:
                    if isinstance(layer, int) or isinstance(layer, float):
                        poisoned_params.append(0)
                    else:
                        poisoned_params.append(np.zeros_like(layer))

            elif self._poisoningMode == "random":
                for layer in current_params:
                    if isinstance(layer, int) or isinstance(layer, float):
                        poisoned_params.append(np.random.random())
                    else:
                        poisoned_params.append(np.random.random_sample(layer.shape))

            else:
                for layer in current_params:
                    poisoned_params.append(layer * -1)

            self._logger.info("Original:")
            for i in current_params:
                if not (isinstance(i, int) or isinstance(i, float)):
                    self._logger.info(i.shape)
                else:
                    self._logger.info(i)

            self._logger.info("Poisoned:")
            for i in poisoned_params:
                if not (isinstance(i, int) or isinstance(i, float)):
                    self._logger.info(i.shape)
                else:
                    self._logger.info(i)

            self._logger.debug(
                "Sending local model parameters to model aggregation server..."
            )

            self._m_aggr_server.send(poisoned_params)
            self._model.set_parameters(poisoned_params)  # new_params)

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

            self._logger.debug("Updating local model with global parameters...")
            self._logger.info("Poisoned local trainings data:")

            # new_params = cast(np.array, m_aggr_msg)
            self._model.set_parameters(poisoned_params)  # new_params)

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
                self._logger.debug("Pay attention, poisoned model overwritten")
                new_params = cast(np.array, m_aggr_msg)
                self._model.set_parameters(new_params)
        else:
            self._logger.debug(
                "AsyncUpdater: Request for sampled federated update received..."
            )
            self.fed_update()
