"""
    A simple server that should collect the evaluation metrics from the clients and process them

    Author: Seraphin Zunzer
    Modified: 09.05.22
"""
import logging
import threading
from datetime import time
from random import random
from time import sleep

from src.communication import EndpointServer
from src.federated_ids_components.dashboard import Dashboard
from typing import Tuple


class FederatedOnlineEvaluator():

    _logger: logging.Logger

    evaluation_objects = []
    _logged_metrics = {'accuracy': {'node_addr_1': [0.1,0.1,0.1,0.1,0.1,0.1], 'node_addr_2':[0.2,0.2,0.2,0.2,0.2,0.2]},
                       'f1': {'node_addr_1': [0.3,0.4,0.1,0.5,0.6,0.9], 'node_addr_2':[0.1,0.9,0.2,0.3,0.5,0.6]},
                        'precision': {'node_addr_1': [0.2, 0.2, 0.3, 0.5, 0.7, 0.9], 'node_addr_2': [0.2, 0.4, 0.5, 0.1, 0.5, 0.6]},
                        'recall': {'node_addr_1': [0.7, 0.6, 0.1, 0.4, 0.9, 0.4], 'node_addr_2': [0.5, 0.2, 0.6, 0.7, 0.9, 0.6]}}


    def __init__(self, addr: tuple[str, int],
                 name: str = ""):
        """ initialize evaluator

        :param name: Name of evaluator for logging purposes.
        :param addr: Address of the evaluator
        """
        self._logger = logging.getLogger(name)
        self._logger.info("Initializing federated evaluator...")

        self._addr = addr
        #self._eval_server = EndpointServer(name="Evaluator", addr=self._addr)
        #self._eval_server.start()

        self.dashboard = Dashboard(self, window_size=None)
        dashbord = threading.Thread(target=self.dashboard.run)
        dashbord.start()

        #Just a simulation to add new metrics
        self.simulate_process()

        while 1:
            r_ready,_ = self._eval_server.poll_connections()
            for i in r_ready.items():
                addr, ep = i
                try:
                   r_metrics = ep.receive()
                   if isinstance(r_metrics, dict):
                       self.queue_metrics(addr, r_metrics)
                   else:
                       self._logger.warning("Received malformed metrics!")
                except RuntimeError:
                    continue
                while True:
                    try:
                       ep.receive(timeout=0)
                    except (TimeoutError, RuntimeError):
                       break

    def queue_metrics(self, addr, recv_metrics):
        """
        Store metrics in dictionary in the right place according to node address.
        TODO: Consider to remove older values
        """
        for metric in recv_metrics:
            if addr in self._logged_metrics[metric]:
                self._logged_metrics[metric][addr] += recv_metrics[metric]
            else:
                self._logged_metrics[metric][addr] = recv_metrics[metric]

    def simulate_process(self):
        """ Test dashboard by adding random values to metrics

        """
        t = 0
        while 1:
            self._logged_metrics['accuracy']['node_addr_1'].append(random())
            self._logged_metrics['f1']['node_addr_1'].append(random())
            self._logged_metrics['precision']['node_addr_1'].append(random())
            self._logged_metrics['recall']['node_addr_1'].append(random())
            self._logged_metrics['accuracy']['node_addr_2'].append(random())
            self._logged_metrics['f1']['node_addr_2'].append(random())
            self._logged_metrics['precision']['node_addr_2'].append(random())
            self._logged_metrics['recall']['node_addr_2'].append(random())
            sleep(2)
            t += 1
            if t % 10 == 0:
                self.queue_metrics("new_node", {"accuracy": [0.4, 0.5, 0.4, 0.5, 0.4, 0.5, 0.4, 0.5, 0.4, 0.5],
                                                "f1": [0.4, 0.5, 0.4, 0.5, 0.4, 0.5, 0.4, 0.5, 0.4, 0.5]})



if __name__ == "__main__":
    eval = FederatedOnlineEvaluator(("127.0.0.1", 54323))

