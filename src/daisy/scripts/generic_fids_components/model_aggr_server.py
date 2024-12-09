# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Pre-configured model aggregation server for a centralized federated system,
whose clients learn cooperatively with each other through this centralized model
aggregation server using the federated averaging (FedAvg) technique. This processing
is done in online manner (as is the general nature of all current federated
processing nodes) and in synchronous fashion, as the server calls upon the clients'
models in periodic manner.

Note this server does nothing on its own --- it requires at least one (in
practicality two) active client to be run in tandem with it, whose set to
asynchronous federated updating, since the server does the initial requests for model
updates.

Author: Fabian Hofmann
Modified: 17.06.24
"""

import argparse
import logging

from daisy.federated_ids_components import FederatedModelAggregator
from daisy.federated_learning import FedAvgAggregator
from daisy.personalized_fl_components.distillative.distillative_model_aggregator import (
    DistillativeModelAggregator,
)


def _parse_args() -> argparse.Namespace:
    """Creates a parser for the server arguments and parses them.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--debug", type=bool, default=False, metavar="", help="Show debug outputs"
    )
    parser.add_argument(
        "--serv",
        default="0.0.0.0",
        metavar="",
        help="IP or hostname of model aggregation server",
    )
    parser.add_argument(
        "--servPort",
        type=int,
        default=8003,
        choices=range(1, 65535),
        metavar="",
        help="Port of model aggregation server",
    )

    aggr_options = parser.add_argument_group("Aggregator Options")
    aggr_options.add_argument(
        "--timeout",
        type=int,
        default=10,
        metavar="",
        help="Timeout to receive model updates from federated clients",
    )
    aggr_options.add_argument(
        "--updateInterval",
        type=int,
        default=None,
        metavar="",
        help="Federated updating step interval, defined by time (s)",
    )
    aggr_options.add_argument(
        "--numClients",
        type=int,
        default=None,
        metavar="",
        help="Number of federated clients to sample during an aggregation step",
    )
    aggr_options.add_argument(
        "--inputSize",
        type=int,
        default=None,
        metavar="",
        help="Input size of dataset required for multi teacher knowledge distillation",
    )
    aggr_options.add_argument(
        "--dashboardURL",
        default="http://localhost:8000",
        metavar="",
        help="IP of (external) dashboard server",
    )
    aggr_options.add_argument(
        "--pflMode",
        type=str,
        choices=["generative", "distillative", "layerwise"],
        default=None,
        metavar="",
        help="Enable personalized FL and choose desired method: generative, distillative, layerwise",
    )

    return parser.parse_args()


def create_server():
    """Creates a pre-configured federated model aggregation server node.
    Entry point of this module's functionality.

    See the header doc string of this module for more details about the preset
    configuration.
    """
    # Args parsing
    args = _parse_args()
    if args.debug:
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.DEBUG,
        )
    else:
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )

    aggr = FedAvgAggregator()

    if args.pflMode == "distillative":
        server = DistillativeModelAggregator(
            m_aggr=aggr,
            addr=(args.serv, args.servPort),
            timeout=args.timeout,
            update_interval=args.updateInterval,
            num_clients=args.numClients,
            dashboard_url=args.dashboardURL,
            input_size=args.inputSize,
        )
    elif args.pflMode == "layerwise":
        raise NotImplementedError
    else:
        server = FederatedModelAggregator(
            m_aggr=aggr,
            addr=(args.serv, args.servPort),
            timeout=args.timeout,
            update_interval=args.updateInterval,
            num_clients=args.numClients,
            dashboard_url=args.dashboardURL,
        )

    server.start()
    input("Press Enter to stop server...")
    server.stop()


if __name__ == "__main__":
    create_server()
