"""
    Class for Federated Aggregator TODO

    Author: Fabian Hofmann
    Modified: 27.10.23
"""
import logging
import threading
from abc import ABC, abstractmethod
from time import time

import numpy as np

from src.communication import EndpointServer
from src.federated_learning import FederatedModel


class FederatedOnlineAggregator(ABC):
    """TODO
    """
    _logger: logging.Logger

    _model: FederatedModel
    _m_lock: threading.Lock

    _aggr_serv: EndpointServer

    _update_interval_t: int
    _t_last_update: float

    _fed_updater: threading.Thread
    _started: bool

    def __init__(self, model: FederatedModel, addr: tuple[str, int],
                 name: str = "", update_interval_t: int = None):
        """TODO

        :param model: Actual model to be fitted and run predictions alternatingly in online manner.
        :param name: Name of federated online node for logging purposes.
        :param addr: TODO
        :param update_interval_t: Federated updating interval, defined by time; every X seconds, do a sync update.
        """
        self._logger = logging.getLogger(name)
        self._logger.info("Initializing federated online node...")

        self._model = model
        self._m_lock = threading.Lock()

        self._aggr_serv = EndpointServer(name="AggrServer", addr=addr, multithreading=True)

        self._update_interval_t = update_interval_t
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

        self._loc_learner = threading.Thread(target=self._create_loc_learner, daemon=True)
        self._loc_learner.start()
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

        self._loc_learner.join()
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

    def _create_loc_learner(self):
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


class FederatedModelAggregator(FederatedOnlineAggregator):
    """TODO

    """
    pass


#
#   Created: 2023
#   Author: Seraphin Zunzer
#   License: MIT License
# ---------------------------------------------------------------


# Optimized server for real-world dataset.
# Federated learning server which starts the federated learning process on the clients,
# receives the newly trained client federated_models,
# and aggregates them using FedAVG.
# FIXME (everything)


import logging
import threading
from time import sleep
from typing import Tuple

from federated_learning.TOBEREMOVEDmodels.autoencoder import FedAutoencoder
from federated_learning.federated_model import FederatedModel
from model_aggregation.FedAvg.fedavg import FedAvg
from model_aggregation.federated_aggregation import FederatedAggregation

import src.communication.message_stream as ms

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class AggregationServer():
    client_queue = []

    def __init__(self, addr: Tuple[str, int], federated_model: FederatedModel,
                 aggregation_method: FederatedAggregation):
        self._addr = addr
        self._model = federated_model
        self._aggregation_method = aggregation_method
        self.client_queue = []

    def recv_registrations(self):
        """
        Receive registration thread function.
        Wait for clients to connect and add client StreamEndpoints to client_queue

        :return: -
        """
        while 1:
            new_client = ms.StreamEndpoint(name="Registration", addr=self._addr, acceptor=True, multithreading=False)
            new_client.start()
            logging.info("Client registered")
            self.client_queue.append(new_client)

    def run(self):
        """Run federated training process:
            - receive registrations
            - send out weights to registered clients
            - receive client weights
            - aggregate weights

        :return: -
        """

        training_thread = threading.Thread(target=self.recv_registrations, name="Registration", daemon=True)
        training_thread.start()
        sleep(10)
        _round = 0
        while 1:
            logging.info(f"START ROUND {_round} ")
            logging.info(f'Registered Clients: {self.client_queue}')

            for client in self.client_queue:
                client.send(self._model._model.get_weights())
                logging.info(f"Started training on client {client}.")

            logging.info("Started training on all available clients.")

            sleep(10)

            for client in self.client_queue:
                logging.info(f"Waiting for weights of client {client}.")
                try:
                    client_weights = client.receive(10)
                    logging.info(f"Received weights from {client}. Start aggregation of weights.")
                    aggregated_weights = self._aggregation_method.aggregate(self._model._model.get_weights())
                    logging.info(f"Set new global weights.")
                    self._model._model.set_weights(aggregated_weights)
                except TimeoutError:
                    logging.warning(f'{client} did not respond.')

            logging.info("Finished training on all available clients.")
            sleep(10)
            _round += 1


# TODO MUST BE OUTSOURCED INTO A PROPER STARTSCRIPT FOR DEMO PURPOSES
if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)

    agg = AggregationServer(("127.0.0.1", 54322), federated_model=FedAutoencoder(), aggregation_method=FedAvg())
    agg.run()
