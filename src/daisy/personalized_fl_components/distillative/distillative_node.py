# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""The distillative aggregation worker node.

Author: Seraphin Zunzer
Modified: 13.01.2025
"""

from time import sleep

import numpy as np
import tensorflow as tf

from daisy.communication import StreamEndpoint
from daisy.data_sources import DataHandler
from daisy.federated_ids_components import FederatedOnlineNode
from daisy.federated_ids_components.node import _try_ops
from daisy.federated_learning import FederatedIFTM, SMAvgTM, FederatedModel
from daisy.personalized_fl_components.auto_model_scaler import AutoModelScaler
from daisy.model_poisoning.model_poisoning import model_poisoning


class pflDistillativeNode(FederatedOnlineNode):
    """Node for learning personalized models using the concept of
    knowledge distillation, as the node also needs to perform knowledge distillation.
    Only difference to a normal node is, that the global model may have a different
    shape. Needs to be started with the corresponding distilative model aggregation
    server to be able to aggregate heterogeneous models
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
        poisoning_mode: str = None,
        input_size: int = 65,
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
        self._poisoningMode = poisoning_mode
        self._input_size = input_size

    def setup(self):
        _try_ops(lambda: self._m_aggr_server.start(), logger=self._logger)

    def cleanup(self):
        _try_ops(lambda: self._m_aggr_server.stop(shutdown=True), logger=self._logger)

    def generate_random_data(self, num_samples, input_shape):
        """
        Fuction to generate normal distributed gaussian samples for knowledge
        distillation process.
        :param num_samples: number of synthetic samples to create.
        :param input_shape: shape of the created synthetic vectors

        :return: array of random data
        """
        return np.random.randn(num_samples, input_shape).astype(np.float32)

    def knowledge_distillation(self, input_data, teacher_model, student_model, epochs):
        """
        Conducts the supervised training of the global model, based on the predictions
        of the received student models from the nodes.

        :param input_data: random input data for creating predictions.
        :param teacher_model: the teacher model to extract knowledge from.
        :param student_model: the student model to distill knowledge in.
        :param epochs: number of training epochs.
        """

        for epoch in range(epochs):
            teacher_logits = teacher_model.predict(input_data)
            self._logger.info(f"Knowledge Distillation {epoch}/{epochs}")
            student_model.fit(input_data, teacher_logits)

    def distillative_aggregation(self, new_params):  # TODO What type is this?
        """
        Single teacher knowledge distillation.
        We initialize the global model, get the predictions on the random dataset,
        and train the student global model in a supervised manner based on these
        predictions. Note that there is no return as there is no need to set the new
        local model weight This is already done by training.

        :param client_models: List of client model weights.  TODO what parameter is this? And where is new_params?
        """

        input_shape = self._input_size
        self._logger.debug("Generate normal distributed random samples")
        x_synthetic = self.generate_random_data(1000, input_shape)
        self._logger.debug("Start distillation of global model into local model")

        # initialize global model
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = tf.keras.losses.MeanAbsoluteError()
        batch_size = 32
        t_m = SMAvgTM()
        err_fn = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        aMS = AutoModelScaler()
        id_fn = aMS.get_manual_model(
            identifier="large",
            input_size=input_shape,
            optimizer=optimizer,
            loss=loss,
            batch_size=batch_size,
            epochs=2,
        )
        teacher_model = FederatedIFTM(
            identify_fn=id_fn, threshold_m=t_m, error_fn=err_fn
        )

        self._logger.debug("Global model initialized with following layers:")
        for i in teacher_model.get_parameters():
            if not (isinstance(i, int) or isinstance(i, float)):
                self._logger.debug(i.shape)
            else:
                self._logger.debug(i)
        self._logger.debug("Received global parameters:")
        for i in new_params:
            if not (isinstance(i, int) or isinstance(i, float)):
                self._logger.debug(i.shape)
            else:
                self._logger.debug(i)

        teacher_model.set_parameters(new_params)
        self._logger.info("Teacher model initialized, starting distillation process")
        self.knowledge_distillation(x_synthetic, teacher_model, self._model, epochs=2)
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
            self._logger.debug("Model Parameters:")
            for i in current_params:
                if not (isinstance(i, int) or isinstance(i, float)):
                    self._logger.debug(i.shape)
                else:
                    self._logger.debug(i)

            params = model_poisoning(
                current_params, self._poisoningMode, self.model
            )  # TODO self.model does not exist!!
            self._m_aggr_server.send(params)

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
                # poisoning mode: random or zeros -> we dont need to update local
                # parameters
                self._logger.debug("Updating local model with global parameters...")
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
                if self._poisoningMode is None or self._poisoningMode == "inverse":
                    # poisoning mode: random or zeros -> we dont need to update local
                    # parameters
                    self._logger.debug("Updating local model with global parameters...")
                    self.distillative_aggregation(m_aggr_msg)

        else:
            self._logger.debug(
                "AsyncUpdater: Request for sampled federated update received..."
            )
            self.fed_update()
