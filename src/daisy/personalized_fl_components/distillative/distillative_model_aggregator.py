# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""An adapted federated online aggregator to be able to aggregate models of different sizes and shapes.
  The models size of a node in pFL is determined by auto_model_scaler.py and collected by the model aggregator.
  The model aggregator uses multi teacher knowledge distillation to exchange knowledge between these models.

Author: Seraphin Zunzer
Modified: 04.07.24
"""

from random import sample
from time import sleep
from typing import cast, Sequence

import numpy as np

from daisy.communication import StreamEndpoint
from daisy.federated_learning import ModelAggregator

from daisy.federated_ids_components import FederatedOnlineAggregator

import tensorflow as tf
from tensorflow import keras

from daisy.personalized_fl_components.auto_model_scaler import AutoModelScaler
from daisy.federated_learning import FederatedIFTM
from daisy.federated_learning import EMAvgTM


class DistillativeModelAggregator(FederatedOnlineAggregator):
    """Centralized federated learning model aggregator using knowledge distillation."""

    _m_aggr: ModelAggregator

    _global_model: keras.Model

    _update_interval: int
    _num_clients: int
    _min_clients: float
    _input_size: int

    def __init__(
        self,
        m_aggr: ModelAggregator,
        addr: tuple[str, int],
        name: str = "",
        timeout: int = 10,
        update_interval: int = None,
        num_clients: int = None,
        min_clients: float = 0.5,
        dashboard_url: str = None,
        input_size: int = None,
    ):
        """Creates a new federated model aggregator. If update_interval is not set,
        defaults to asynchronous federated model aggregation, i.e. waiting for
        individual clients to report their local models in unspecified/unset
        intervals, before sending them the freshly aggregated global model back.

        :param m_aggr: Actual aggregator to aggregate models with.
        :param addr: Address of aggregation server for clients to connect to.
        :param name: Name of federated model aggregator for logging purposes.
        :param timeout: Timeout for waiting to receive local model updates from
        federated clients.
        :param update_interval: Sets how often the aggregation server should do a (
        synchronous) federated aggregation step to compute a global model to send to
        the clients (in seconds).
        :param num_clients: Allows sampling of client population to perform a sampled
        synchronous aggregation step. If none provided, always attempts to request
        models from the entire population. For async mode, threshold of waiting
        clients for a model update.
        :param min_clients: Minimum ratio of responsive clients after a request for
        models during a sampled synchronous aggregation step, aborts aggr step if not
        satisfied. If none provided, tolerates a 50% failure rate.
        :param dashboard_url: URL to dashboard to report aggregation statics to.
        """
        super().__init__(
            addr=addr, name=name, timeout=timeout, dashboard_url=dashboard_url
        )

        self._m_aggr = m_aggr

        self._update_interval = update_interval
        self._num_clients = num_clients
        self._min_clients = min_clients
        self._input_size = input_size
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name="train_acc")

        # we define the global model to have this structure
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = tf.keras.losses.MeanAbsoluteError()

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
        self._global_model = FederatedIFTM(
            identify_fn=id_fn, threshold_m=t_m, error_fn=err_fn
        )

    def setup(self):
        pass

    def cleanup(self):
        pass

    def generate_random_data(self, num_samples, input_shape):
        """
        Fuction to generate normal distributed gaussian samples for knowledge distillation process.
        """
        return np.random.random((num_samples, input_shape)).astype(np.float32)

    # return np.random.normal((num_samples, input_shape)).astype(np.float32)
    # TODO check random.normal /random random for gaussian
    #        return np.random.random((num_samples, input_shape)).astype(np.float32)

    def knowledge_distillation(self, input_data, teacher_model, student_model, epochs):
        """
        Conducts the supervised training of the global model, based on the predictions of the received
        student models from the nodes.

        """
        for epoch in range(epochs):
            self._logger.info(f"Knowledge Distillation {epoch}/{epochs}")
            teacher_logits = teacher_model.predict(input_data)
            student_model.fit(input_data, teacher_logits)

    def mtkd(self, client_models):
        """
        Multi teacher knowledge distillation.

        """
        num_samples = 1000

        self._logger.info("Generate gaussian samples")
        X_synthetic = self.generate_random_data(num_samples, self._input_size)
        self._logger.info(f"Start MTKD of {len(client_models)} teachers")

        for model_params in client_models:
            # For evaluation, we assume all nodes have the following model structure
            # when implementing uuids, the server could look up the to the uuid corresponding model structure

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            loss = tf.keras.losses.MeanAbsoluteError()
            batchSize = 32
            t_m = EMAvgTM()
            epochs = 3
            err_fn = tf.keras.losses.MeanAbsoluteError(
                reduction=tf.keras.losses.Reduction.NONE
            )
            aMS = AutoModelScaler()
            id_fn = aMS.get_manual_model(
                "medium", self._input_size, optimizer, loss, batchSize, epochs
            )
            teacher_model = FederatedIFTM(
                identify_fn=id_fn, threshold_m=t_m, error_fn=err_fn
            )

            self._logger.info("Teacher model initialized with following layers:")
            for i in teacher_model.get_parameters():
                if not (isinstance(i, int) or isinstance(i, float)):
                    self._logger.info(i.shape)
                else:
                    self._logger.info(i)

            self._logger.info("Received teacher parameters:")
            for i in model_params:
                if not (isinstance(i, int) or isinstance(i, float)):
                    self._logger.info(i.shape)
                else:
                    self._logger.info(i)

            teacher_model.set_parameters(model_params)
            self._logger.info("Teacher initialized, starting distillation process")

            self.knowledge_distillation(
                X_synthetic, teacher_model, self._global_model, epochs
            )

            self._logger.info("Continue with next teacher")

        self._logger.info("multi Teacher distillation finished")
        return self._global_model.get_parameters()

    def create_fed_aggr(self):
        """Starts the loop to either synchronously aggregate in fixed intervals or
        check in looping manner whether there are federated clients requesting an
        asynchronous updating step.
        """
        self._logger.info("Starting model aggregation loop...")
        while self._started:
            try:
                if self._update_interval is not None:
                    self._logger.debug(
                        "Initiating interval-based synchronous aggregation step..."
                    )
                    self._sync_aggr()
                    sleep(self._update_interval)
                else:
                    self._logger.debug(
                        "Checking federated clients for "
                        "asynchronous aggregation requests..."
                    )
                    self._async_aggr()

                self._update_dashboard(
                    "/aggregation/",
                    {
                        "agg_status": "Operational",
                        "agg_count": len(
                            self._aggr_serv.poll_connections()[1].values()
                        ),
                        "agg_nodes": str(self._aggr_serv.poll_connections()[1].keys()),
                    },
                )
            except RuntimeError:
                # stop() was called
                break
        self._logger.info("Model aggregation loop stopped.")

    def _sync_aggr(self):
        """Performs a synchronous, sampled federated aggregation step, i.e. sending
        out a request for local models to a sample of the client population. If
        sufficient clients respond to the request after a set timeout, their models
        are aggregated, potentially also taking into account the global model of the
        previous step (depending on the model aggregator chosen), and the newly
        created global model is sent back to all available clients.
        """
        clients = self._aggr_serv.poll_connections()[1].values()
        if len(clients) == 0:
            self._logger.info("No clients available for aggregation step!")
            return
        if self._num_clients is not None:
            if len(clients) < self._num_clients:
                self._logger.info(
                    f"Insufficient write-ready clients [{len(clients)}] "
                    f"available for aggregation step [{self._num_clients}]!"
                )
                return
            clients = sample(cast(Sequence, clients), self._num_clients)
        num_clients = len(clients)

        self._logger.debug(
            f"Requesting local models from sampled clients [{len(clients)}]..."
        )
        for client in clients:
            client.send(None)
        sleep(self._timeout)

        clients = StreamEndpoint.select_eps(clients)[0]
        self._logger.debug(
            "Receiving local models from available "
            f"requested clients [{len(clients)}]..."
        )
        client_models = [
            model
            for model in cast(
                list[list[np.ndarray]],
                StreamEndpoint.receive_latest_ep_objs(clients, list).values(),
            )
            if model is not None
        ]

        if len(client_models) < min(1, int(num_clients * self._min_clients)):
            self._logger.info(
                f"Insufficient number of client models [{len(client_models)}] "
                f"for aggregation received [{self._num_clients}]!"
            )
            return
        self._logger.debug(
            f"Aggregating client models [{len(client_models)}] into global model..."
        )
        self._logger.info("Start distillative aggregation")
        aggregated_params = self.mtkd(client_models)

        clients = self._aggr_serv.poll_connections()[1].values()
        self._logger.info(
            f"Sending aggregated global model to all "
            f"available clients [{len(clients)}]..."
        )
        for i in aggregated_params:
            if not (isinstance(i, int) or isinstance(i, float)):
                self._logger.info(i.shape)
            else:
                self._logger.info(i)
        self._logger.info("\n")
        for client in clients:
            client.send(aggregated_params)

        self._logger.info("Sending aggregated global model to nodes ")

        self._update_dashboard(
            "/aggregation/",
            {
                "agg_status": "Operational",
                "agg_count": len(self._aggr_serv.poll_connections()[1].values()),
            },
        )

    def _async_aggr(self):
        """Performs an asynchronous federated aggregation step, i.e. checking whether
        a sufficient number of clients sent their local models to the server,
        before aggregating their models, potentially also taking into account the
        global model of the previous step (depending on the model aggregator chosen).
        The newly created global model is sent back only to all clients that
        requested an aggregation step.
        """
        clients = self._aggr_serv.poll_connections()[0].values()
        if len(clients) == 0:
            self._logger.info("No clients available for aggregation step!")
            sleep(1)
            return
        if self._num_clients is not None and len(clients) < self._num_clients:
            self._logger.info(
                f"Insufficient read-ready clients [{len(clients)}] "
                f"available for aggregation step [{self._num_clients}]!"
            )
            sleep(1)
            return

        self._logger.debug(
            f"Receiving local models from requesting clients [{len(clients)}]..."
        )
        client_models = [
            model
            for model in cast(
                list[list[np.ndarray]],
                StreamEndpoint.receive_latest_ep_objs(clients, list).values(),
            )
            if model is not None
        ]

        if len(client_models) == 0 or (
            self._num_clients is not None and len(client_models) < self._num_clients
        ):
            self._logger.info(
                f"Insufficient number of client models [{len(client_models)}] "
                f"for aggregation received [{self._num_clients}]!"
            )
            sleep(1)
            return
        self._logger.debug(
            f"Aggregating client models [{len(client_models)}] into global model..."
        )
        global_model = self._m_aggr.aggregate(client_models)

        clients = StreamEndpoint.select_eps(clients)[1]
        self._logger.debug(
            "Sending aggregated global model to available "
            f"requesting clients [{len(clients)}]..."
        )
        for client in clients:
            client.send(global_model)

        self._update_dashboard(
            "/aggregation/",
            {
                "agg_status": "Operational",  # TODO add len(client_models)
                "agg_count": len(clients),
            },
        )
