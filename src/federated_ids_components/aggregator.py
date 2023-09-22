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
