#
#   Created: 2023
#   Author: Seraphin Zunzer
#   License: MIT License
# ---------------------------------------------------------------


# Optimized server for real-world dataset.
# Federated learning server which starts the federated learning process on the clients,
# receives the newly trained client models,
# and aggregates them using FedAVG.


import logging
from typing import Tuple

import src.communication.message_stream as ms
from src.federated_ids import federated_model as fm

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


class aggregation_server():
    client_queue = []

    class registration_thread()

        def __init__(self, addr: Tuple[str, int]):
            self._addr = addr
            threading.Thread.__init__(self)

        def recv_registrations(self):
            """
            Receive registration messages on startup

            :param s:
            :return:
            """
            while 1:
                new_client = ms.StreamEndpoint(self._addr)
                client_queue.append[new_client]

    def __init__(self, addr: Tuple[str, int], federated_model: fm.FederatedModel):
        self._addr = addr
        self.model = federated_model.create_model()

    def start_client_training(client: ms.StreamEndpoint):
        """
        Connect to client, send global weights which start the training

        :param client_port: Port of the client to send the data to
        :param server_weights_data: List of the latest server model weights
        :return: -
        """
        client.send(self.model.get_weights())

    def client_response(self, client):
        """
        Receive updated model weights

        :param s: Socket to receive data on
        :return: Multiple variables, none if it is a no data or registration message, the client data and metrics otherwise
        """
        averaged_sum(client.recv())

    def averaged_sum(self, client_weights):
        """
        Calculate fedavg between global weights and new weights

        :param global_weights:
        :param client_weights:
        :return:
        """
        try:
            global_weights = self.model.get_weights()
            for i in range(0, len(global_weights)):
                for j in range(0, len(global_weights[i])):
                    global_weights[i][j] = (global_weights[i][j] + client_weights[i][j]) / 2

            self.model.set_weights(global_weights)
        except:
            print("Malformed weights received!")

    def federated_training(self):
        """Conduct federated training, send out weigths to clients and receive local weights

        :return:-
        """

        t = Thread(target=registration_thread, name="Registration", daemon=True)
        t.open()

        _round = 0
        while 1:
            logging.info(f"START ROUND {_round} ")

            for client in client_queue:
                start_client_training(client)

            logging.info("Started training on all available clients.\n")

            for client in client_queue:
                client_response(client)

            logging.info("Finished training on all available clients.\n")
