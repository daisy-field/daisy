# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""TODO Docstring for server

    Pre-configured demonstration client for a federated intrusion detection system (IDS), that learns cooperatively with
    another clients through a centralized model aggregation server using the federated averaging (FedAvg) technique. In
    this example, the client is configured to process network traffic data from the road-side infrastructure (BeIntelli)
    on Cohda boxes 2 and 5 on March 6th 2023.

    This processing is done in online manner (as is the general nature of all current federated processing nodes), with
    the underlying model running predictions on a minibatch, before training a single epoch on that batch. The model
    itself is a hybrid approach for anomaly detection, using a simple autoencoder paired with a dynamic threshold to map
    the anomaly score to a binary label. Finally, the prediction results are evaluated using a sliding window confusion
    matrix along its anomaly detection evaluation metrics (e.g. Precision, Recall, F1-score, etc.).

    Note that this demonstration client can also be launched as a standalone detection component, if no additional
    client is run along with the model aggregation server. The same is the case for additional prediction and evaluation
    result aggregation using centralize servers (see -h for more information).

    Author: Fabian Hofmann
    Modified: 30.01.24
"""

import argparse
import logging

from daisy.federated_ids_components import FederatedModelAggregator
from daisy.federated_learning import FedAvgAggregator


def _parse_args() -> argparse.Namespace:
    """Creates a parser for the server arguments and parses them.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--debug", type=bool, default=False,
                        metavar="", help="Show debug outputs")
    parser.add_argument("--serv", required=True,
                        help="IP or hostname of model aggregation server")
    parser.add_argument("--servPort", type=int, default=8001, choices=range(1, 65535),
                        metavar="", help="Port of model aggregation server")

    aggr_options = parser.add_argument_group("Aggregator Options")
    aggr_options.add_argument("--timeout", type=int, default=10,
                              metavar="", help="Timeout to receive model updates from federated clients")
    aggr_options.add_argument("--updateInterval", type=int, default=None,
                              metavar="", help="Federated updating interval, defined by time (s)")

    return parser.parse_args()


def create_server():
    """TODOCreates a pre-configured federated client with preset components that runs on either of the two subsets of the
    March 6th 2023 network traffic data set. Entry point of this module's functionality.

    See the header doc string of this module for more details about the preset client's configuration.
    """
    # Args parsing
    args = _parse_args()
    if args.debug:
        logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                            level=logging.DEBUG)
    else:
        logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                            level=logging.INFO)

    # Aggregator
    aggr = FedAvgAggregator()
    FederatedModelAggregator(m_aggr=aggr, addr=(args.serv, args.servPort),
                             timeout=args.timeout, update_interval=args.update_interval, num_clients=2).start()


if __name__ == "__main__":
    create_server()
