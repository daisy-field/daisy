"""
    A simple server that should collect the evaluation metrics from the clients and process them

    Author: Seraphin Zunzer
    Modified: 09.05.22
"""
# FIXME (everything)
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
    _logged_metrics = {'accuracy': {'node_addr_1': [0.1,0.1,0.1,0.1,0.1,0.1], 'node_addr_2':[0.2,0.2,0.2,0.2,0.2,0.2], 'node_new':[]},
                       'f1': {'node_addr_1': [0.3,0.4,0.1,0.5,0.6,0.9], 'node_addr_2':[0.1,0.9,0.2,0.3,0.5,0.6], 'node_new':[]},
                        'precision': {'node_addr_1': [0.2, 0.2, 0.3, 0.5, 0.7, 0.9], 'node_addr_2': [0.2, 0.4, 0.5, 0.1, 0.5, 0.6], 'node_new':[]},
                        'recall': {'node_addr_1': [0.7, 0.6, 0.1, 0.4, 0.9, 0.4], 'node_addr_2': [0.5, 0.2, 0.6, 0.7, 0.9, 0.6], 'node_new':[]}}


    def __init__(self, addr: tuple[str, int],
                 name: str = ""):
        """

        :param model: Actual model to be fitted and run predictions alternatingly in online manner.
        :param name: Name of federated online node for logging purposes.
        :param addr:
        :param update_interval_t: Federated updating interval, defined by time; every X seconds, do a sync update.
        """
        self._logger = logging.getLogger(name)
        self._logger.info("Initializing federated evaluator node...")

        self._addr = addr
        #self._eval_server = EndpointServer(name="Evaluator", addr=self._addr)
        #self._eval_server.start()
        self.evaluation_objects = []

        self.dashboard = Dashboard(self)
        dashbord = threading.Thread(target=self.dashboard.run)
        dashbord.start()

        #Just a simulation to add new metrics
        while 1:
            self._logged_metrics['accuracy']['node_addr_1'].append(random())
            self._logged_metrics['f1']['node_addr_1'].append(random())
            self._logged_metrics['precision']['node_addr_1'].append(random())
            self._logged_metrics['recall']['node_addr_1'].append(random())
            self._logged_metrics['accuracy']['node_addr_2'].append(random())
            self._logged_metrics['f1']['node_addr_2'].append(random())
            self._logged_metrics['precision']['node_addr_2'].append(random())
            self._logged_metrics['recall']['node_addr_2'].append(random())
            sleep(2.4)

        while not 1:
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
        TODO: Check if client exists, if not create empty list before
        """
        for metric in recv_metrics:
            self._logged_metrics[metric][addr].append(recv_metrics[metric])






if __name__ == "__main__":
    eval = FederatedOnlineEvaluator(("127.0.0.1", 54323))

