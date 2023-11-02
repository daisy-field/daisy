"""
    A simple server that should collect the evaluation metrics from the clients and process them

    Author: Seraphin Zunzer
    Modified: 09.05.22
"""
# FIXME (everything)
import logging

#from communication import StreamEndpoint
from src.federated_ids_components.dashboard import Dashboard
from src.federated_ids_components.dashboard import testDashboard
from typing import Tuple


class FederatedOnlineEvaluator():

    _logger: logging.Logger

    evaluation_objects = []

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
        self._eval_server = StreamEndpoint(name="Evaluator", addr=self._addr)
        self._eval_server.start()
        self.evaluation_objects = []
        while 1:
            self.evaluation_objects.append(self._eval_server.receive())


if __name__ == "__main__":
    #eval = FederatedOnlineEvaluator(("127.0.0.1", 54323))

    dashboard = testDashboard()
    dashboard.start()
