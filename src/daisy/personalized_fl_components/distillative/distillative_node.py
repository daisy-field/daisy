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

import logging
from time import sleep
from typing import Callable, cast

import numpy as np
import tensorflow as tf

from daisy.communication import StreamEndpoint
from daisy.data_sources import DataHandler
from daisy.federated_learning import FederatedModel

from daisy.federated_ids_components import FederatedOnlineNode

from daisy.personalized_fl_components.auto_model_scaler import AutoModelScaler

from daisy.federated_learning import FederatedIFTM, EMAvgTM


class pflDistillativeNode(FederatedOnlineNode):
    """Node for knowledge distillation. Only difference is, that more model information has to be sent to the model
    aggregation server, and that the global model has to be distilled into the local model.
    """

    _m_aggr_server: StreamEndpoint
    _timeout: int

    def __init__(
        self,
        data_handler: DataHandler,
        batch_size: int,
        model: FederatedModel,
        m_aggr_server: tuple[str, int],
        timeout: int = 60,
        name: str = "",
        label_split: int = 2**32,
        supervised: bool = False,
        metrics: list[tf.metrics.Metric] = None,
        eval_server: tuple[str, int] = None,
        aggr_server: tuple[str, int] = None,
        sync_mode: bool = True,
        update_interval_s: int = None,
        update_interval_t: int = None,
    ):
        """Creates a new federated online client.

        :param data_source: Data source of data stream to draw data points from.
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
            data_handler=data_handler,
        )

        self._m_aggr_server = StreamEndpoint(
            name="MAggrServer",
            remote_addr=m_aggr_server,
            acceptor=False,
            multithreading=True,
        )
        self._timeout = timeout

    def setup(self):
        _try_ops(lambda: self._m_aggr_server.start(), logger=self._logger)

    def cleanup(self):
        _try_ops(lambda: self._m_aggr_server.stop(shutdown=True), logger=self._logger)

    def generate_random_data(self, num_samples, input_shape):
        return np.random.random((num_samples, input_shape)).astype(np.float32)

    def knowledge_distillation(self, images, teacher_model, student_model, epochs):
        for epoch in range(epochs):
            self._logger.info("Get teacher predictions")
            teacher_logits = teacher_model.predict(images)

            self._logger.info("Train student model according to teacher predictions")
            student_model.fit(images, teacher_logits)

    def distillative_aggregation(self, new_params):
        # TODO Seraphin:
        # 1. initialize global model
        # 2. set weights of global model by new_params
        # 3. compile global model
        # 4. create random data and make predictions of global model
        # 5. train local model using these predictions and distillation loss

        num_samples = 1000
        input_shape = 65
        self._logger.info("Node starts generating random samples")
        X_synthetic = self.generate_random_data(num_samples, input_shape)
        self._logger.info("Start distillation of global model into local model")

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = tf.keras.losses.MeanAbsoluteError()
        input_size = 65
        batchSize = 32
        t_m = EMAvgTM()
        epochs = 10
        err_fn = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        aMS = AutoModelScaler()
        id_fn = aMS.get_manual_model(
            "large", input_size, optimizer, loss, batchSize, epochs
        )
        teacher_model = FederatedIFTM(
            identify_fn=id_fn, threshold_m=t_m, error_fn=err_fn
        )

        for i in new_params:
            if not (isinstance(i, int) or isinstance(i, float)):
                self._logger.info(i.shape)
            else:
                self._logger.info(i)
        self._logger.info("\n")

        for i in self._model.get_parameters():
            if not (isinstance(i, int) or isinstance(i, float)):
                self._logger.info(i.shape)
            else:
                self._logger.info(i)

        teacher_model.set_parameters(new_params)
        self._logger.info("Global model initialized, starting distillation process")
        self.knowledge_distillation(X_synthetic, teacher_model, self._model, epochs)
        self._logger.info("Distillation finished")

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
            self._logger.debug(
                "Sending local model parameters to model aggregation server..."
            )
            # TODO Seraphin add identifier for the current selected model, and maybe even client id
            self._m_aggr_server.send(current_params)

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

            self._logger.info("Updating local model with global parameters...")
            # new_params = cast(np.array, m_aggr_msg)

            # TODO Seraphin distill global model into local model
            self.distillative_aggregation(m_aggr_msg)

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
                # TODO Apply distillation to get knowledge from global model into client
                new_params = cast(np.array, m_aggr_msg)
                self._model.set_parameters(new_params)
        else:
            self._logger.debug(
                "AsyncUpdater: Request for sampled federated update received..."
            )
            self.fed_update()


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
