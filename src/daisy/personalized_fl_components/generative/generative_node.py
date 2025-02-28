# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""A generative worker node, using the federated GAN to add synthetic data to the local
model training.

Author: Seraphin Zunzer
Modified: 13.01.25
"""

from time import sleep, time
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from daisy.communication import StreamEndpoint
from daisy.data_sources import DataHandler
from daisy.federated_ids_components import FederatedOnlineNode
from daisy.federated_ids_components.node import _try_ops
from daisy.federated_learning import FederatedModel
from daisy.personalized_fl_components.generative.generative_model import GenerativeModel
from daisy.model_poisoning.model_poisoning import model_poisoning


class pflGenerativeNode(FederatedOnlineNode):
    """Implementation of a federated online node, using a generative model
    for the knowledge exchange in personalized federated learning.
    In pFL, the local nodes model differ. Therefore, a generative model is trained and
    exchanged in a federated manner,
    to locally create augmented data of other nodes."""

    _m_aggr_server: StreamEndpoint
    _timeout: int
    round: int
    name = ""

    def __init__(
        self,
        data_handler: DataHandler,
        batch_size: int,
        model: FederatedModel,
        generative_model: GenerativeModel,
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
        """Creates a new federated online client for data augmentation. Now,
        additionally to the local node, a generative model is stored in the node.

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
        self._generative_model = generative_model
        self.name = name
        self._m_aggr_server = StreamEndpoint(
            name="MAggrServer",
            remote_addr=m_aggr_server,
            acceptor=False,
            multithreading=True,
        )
        self._timeout = timeout
        self._poisoningMode = poisoning_mode
        self.round = 0
        self.scaler = MinMaxScaler()

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
            current_params = self._generative_model.get_parameters()
            self._logger.info("Retreived parameters of GAN")
            for i in current_params:
                if not (isinstance(i, int) or isinstance(i, float)):
                    self._logger.debug(i.shape)
                else:
                    self._logger.debug(i)

            params = model_poisoning(current_params, self._poisoningMode, self._generative_model)
            self._m_aggr_server.send(params)

            self._logger.info(
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
                else:
                    self._logger.warning(
                        "Received model parameters from aggregation server"
                    )
            except TimeoutError:
                self._logger.warning(
                    "Timeout triggered. No message received "
                    "from model aggregation server!"
                )
                return

            # Set local parameters to global parameters
            if self._poisoningMode is None or self._poisoningMode == "inverse":
                self._logger.debug("Updating local GAN with global parameters...")
                new_params = cast(np.array, m_aggr_msg)
                self._generative_model.set_parameters(new_params)

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
                self._generative_model.set_parameters(new_params)
        else:
            self._logger.debug(
                "AsyncUpdater: Request for sampled federated update received..."
            )
            self.fed_update()

    def _process_batch(self):
        """Processes the current batch for both running a prediction and fitting the
        generative model around the databatch. Augmented data from generative model is
        added to training batch used for training the local intrusion detection model.
        Also sends results to both the aggregation and the
        evaluation server, if available and provided in the beginning,
        before flushing the minibatch window.
        """
        self._logger.debug("AsyncLearner: Arranging full minibatch for processing...")
        x_data, y_true = (
            np.array(self._minibatch_inputs),
            np.array(self._minibatch_labels),
        )

        # self.create_heatmap(x_data,"", f"normal_{self.round}_{self.name}")

        self._logger.info("Created synthetic data, start training GAN...")
        self.scaler.fit(x_data)
        scaled = self.scaler.transform(x_data)

        self._generative_model.fit(scaled)
        self._logger.info("Finished training GAN...")

        self._logger.debug("AsyncLearner: Processing minibatch...")
        y_pred = self._model.predict(x_data)
        self._logger.debug(
            f"AsyncLearner: Pr" f"ediction results for minibatch: {(x_data, y_pred)}"
        )
        if self._aggr_serv is not None:
            self._aggr_serv.send((x_data, y_pred))
        if len(self._metrics) > 0:
            eval_res = {metric.name: metric(y_true, y_pred) for metric in self._metrics}
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

        augmented_data = self._generative_model.create_synthetic_data(n=len(x_data))
        upscaled = self.scaler.inverse_transform(augmented_data)

        train_data = tf.concat([x_data, upscaled], axis=0)
        self._logger.info("Merged Augmented and Real Data")

        # self.create_heatmap(upscaled,"", f"synthetic_{self.round}_{self.name}")

        self.round += 1
        self._model.fit(train_data)
        self._logger.info("Prediction model trained")
        self._logger.debug("AsyncLearner: Minibatch processed, cleaning window ...")
        self._minibatch_inputs, self._minibatch_labels = [], []

    def sync_fed_update_check(self):
        """Checks whether the conditions for a synchronous federated update step are
        met and performs it.
        """
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

    def create_heatmap(self, data, path: str, name: str):
        """
        Create a heatmap based of attributes data. Image ist saved under the given name
        at the given path. Useful to compare synthetic data correlations with real data
        correlations.

        :param data: dataset to create the heatmap from.
        :param path: location to store the image.
        :param name: title of the image, e.g. current number of iterations
        """
        df = pd.DataFrame(data)
        correlation_matrix = df.corr()
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", fmt=".2f")
        plt.title(name)
        fig.savefig(path + name, dpi=fig.dpi)
