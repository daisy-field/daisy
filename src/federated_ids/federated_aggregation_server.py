#
#   Created: 2023
#   Author: Seraphin Zunzer
#   License: MIT License
# ---------------------------------------------------------------


# Optimized server for real-world dataset.
# Federated learning server which starts the federated learning process on the clients,
# receives the newly trained client federated_models,
# and aggregates them using FedAVG.


import logging
import threading
from time import sleep
from typing import Tuple

import src.communication.message_stream as ms
from federated_models.federated_model import FederatedModel
from federated_models.models.autoencoder import FedAutoencoder

from model_aggregation.federated_aggregation import FederatedAggregation
from model_aggregation.FedAvg.fedavg import FedAvg

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)



class AggregationServer():
    client_queue = []

    def __init__(self, addr: Tuple[str, int], federated_model: FederatedModel, aggregation_method: FederatedAggregation):
        self._addr = addr
        self._model = federated_model
        self._aggregation_method = aggregation_method
        self.client_queue=[]

    def recv_registrations(self):
            """
            Receive registration thread function.
            Wait for clients to connect and add client StreamEndpoints to client_queue

            :return: -
            """
            while 1:
                new_client = ms.StreamEndpoint(name="ReceivingEndpoint", addr=self._addr)
                new_client.start()
                logging.info("Client registered")
                self.client_queue.append(new_client)
                sleep(30)

    def start_client_training(self, client: ms.StreamEndpoint):
        """
        Connect to client amd send global weights to start the local training

        :param client: Clients StreamEndpoint
        :return: -
        """
        client.send(self._model.get_model_weights())

    def client_response(self, client: ms.StreamEndpoint):
        """
        Receive updated model weights

        :param client: Socket to receive data on
        :return: received model weights
        """
        return client.receive()

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

        _round = 0
        while 1:
            logging.info(f"START ROUND {_round} ")

            for client in self.client_queue:
                self.start_client_training(client)
            logging.info(f'Registered Clients: {self.client_queue}')
            logging.info("Started training on all available clients.\n")

            for client in self.client_queue:
                client_weights = self.client_response(client)
                aggregates_weights = self._aggregation_method.aggregate(self._model.get_model_weights(), client_weights)
                self._model.set_model_weights(aggregates_weights)

            logging.info("Finished training on all available clients.\n")
            _round += 1
            sleep(30)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)

    agg = AggregationServer(("127.0.0.1", 54322), federated_model=FedAutoencoder(), aggregation_method = FedAvg())
    agg.run()

