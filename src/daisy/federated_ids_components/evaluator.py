# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
    A simple server that should collect the evaluation metrics from the clients and process them

    Author: Seraphin Zunzer
    Modified: 09.05.22

    TODO: CLEANUP IN WAY SIMILAR TO AGGREGATOR.PY, JUST ANOTHER AGGREGATOR REALLY (INHERIT FROM BASE AGGR)
"""
import logging
import threading
from datetime import datetime
from random import random
from time import sleep

from daisy.federated_ids_components.dashboard import Dashboard


class FederatedOnlineEvaluator():
    _logger: logging.Logger

    evaluation_objects = []
    _logged_metrics = {
        'accuracy': {'node_addr_1': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 'node_addr_2': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]},
        'f1': {'node_addr_1': [0.3, 0.4, 0.1, 0.5, 0.6, 0.9], 'node_addr_2': [0.1, 0.9, 0.2, 0.3, 0.5, 0.6]},
        'precision': {'node_addr_1': [0.2, 0.2, 0.3, 0.5, 0.7, 0.9], 'node_addr_2': [0.2, 0.4, 0.5, 0.1, 0.5, 0.6]},
        'recall': {'node_addr_1': [0.7, 0.6, 0.1, 0.4, 0.9, 0.4], 'node_addr_2': [0.5, 0.2, 0.6, 0.7, 0.9, 0.6]},
        'x': ['07:41:19', '07:41:20', '07:41:21', '07:41:22', '07:41:23', '07:41:24']}
    _nodes = ["node_addr_1", "node_addr_2"]
    _metrics_to_log = ['accuracy', 'f1', 'recall', 'precision']

    def __init__(self, addr: tuple[str, int],
                 name: str = ""):
        """ initialize evaluator

        :param name: Name of evaluator for logging purposes.
        :param addr: Address of the evaluator
        """
        self._logger = logging.getLogger(name)
        self._logger.info("Initializing federated evaluator...")

        self._addr = addr
        # self._eval_server = EndpointServer(name="Evaluator", addr=self._addr)
        # self._eval_server.start()

        self.dashboard = Dashboard(self, window_size=None)
        dashbord = threading.Thread(target=self.dashboard.run)
        dashbord.start()

        # Just a simulation to add new metrics
        self.simulate_process()

        while 1:
            sleep(10)
            r_nodes = []
            r_ready, _ = self._eval_server.poll_connections()
            for i in r_ready.items():
                addr, ep = i
                if addr not in self._nodes:
                    self.add_node(addr)
                r_nodes.append(addr)
                new_metrics = []
                try:
                    r_metrics = ep.receive()
                    if isinstance(r_metrics, dict):
                        new_metrics.append(r_metrics)
                    else:
                        self._logger.warning("Received malformed metrics!")
                except RuntimeError:
                    continue
                while True:
                    try:
                        r_metrics = ep.receive(timeout=0)
                        if isinstance(r_metrics, dict):
                            new_metrics.append(r_metrics)
                        else:
                            self._logger.warning("Received malformed metrics!")
                    except (TimeoutError, RuntimeError):
                        break
                self.queue_metrics(addr, new_metrics[-1])
            self.fill_empty_metrics(responding_nodes=r_nodes)

    def fill_empty_metrics(self, responding_nodes):
        for addr in self._nodes:
            if addr not in responding_nodes:
                for metric in self._metrics_to_log:
                    self._logged_metrics[metric][addr] += None

    def add_node(self, address):
        """
        Add a new node by noneing all previous values
        """
        for metric in self._metrics_to_log:
            self._logged_metrics[metric][address] = [None] * len(self._logged_metrics['x'])

    def queue_metrics(self, addr, recv_metrics):
        """
        Store metrics in dictionary in the right place according to node address.
        Append none, if metric wasn't received
        """
        for metric in self._metrics_to_log:
            if addr in self._logged_metrics[metric]:
                if metric in recv_metrics:
                    self._logged_metrics[metric][addr] += recv_metrics[metric]
                else:
                    self._logged_metrics[metric][addr] += [None]

    def simulate_process(self):
        """ Test dashboard by adding random values to metrics

        """
        t = 0
        while 1:
            sleep(2)
            self._logged_metrics['x'].append(datetime.now().strftime("%H:%M:%S"))
            self._logged_metrics['accuracy']['node_addr_1'].append(random())
            self._logged_metrics['f1']['node_addr_1'].append(random())
            self._logged_metrics['precision']['node_addr_1'].append(random())
            self._logged_metrics['recall']['node_addr_1'].append(random())
            self._logged_metrics['accuracy']['node_addr_2'].append(random())
            self._logged_metrics['f1']['node_addr_2'].append(random())
            self._logged_metrics['precision']['node_addr_2'].append(random())
            self._logged_metrics['recall']['node_addr_2'].append(random())
            responding_nodes = ['node_addr_1', 'node_addr_2']
            t += 1
            if t % 10 == 0:
                responding_nodes.append("new_node_" + str(t))
                self.add_node("new_node_" + str(t))
                self.queue_metrics("new_node_" + str(t), {"accuracy": [random(), random(), random()],
                                                          "f1": [random(), random(), random()]})

            self.fill_empty_metrics(responding_nodes)


if __name__ == "__main__":
    eval = FederatedOnlineEvaluator(("127.0.0.1", 54323))
