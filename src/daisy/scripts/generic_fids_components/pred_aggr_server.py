# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Pre-configured prediction aggregation server for a centralized federated system,
whose clients learn cooperatively with each other through this centralized model
aggregation server using the federated averaging (FedAvg) technique. This processing
is done in online manner (as is the general nature of all current federated
processing nodes) and in synchronous fashion, as the server calls upon the clients'
models in periodic manner.

Note this server does nothing on its own --- it requires at least one (in
practicality two) active client to be run in tandem with it, whose set to
asynchronous federated updating, since the server does the initial requests for model
updates.

Author: Fabian Hofmann, Seraphin Zunzer
Modified: 12.06.24
"""

import argparse
import logging

from daisy.federated_ids_components import FederatedPredictionAggregator

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
        "--serv", required=True, help="IP or hostname of prediction aggregation server"
    )
    parser.add_argument(
        "--servPort",
        type=int,
        default=8000,
        choices=range(1, 65535),
        metavar="",
        help="Port of prediction aggregation server",
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
        "--window_size",
        type=int,
        default=None,
        metavar="",
        help="Window_size of aggregation",
    )
    aggr_options.add_argument(
        "--dashboard_url",
        default="127.0.0.1",
        help="IP of dashboard server"
    )
    aggr_options.add_argument(
        "--name",
        default="Evaluation Server",
        help="Name of prediction server"
    )

    return parser.parse_args()

def create_server():
    """Creates a pre-configured federated server node for two the federated demo
    clients. Entry point of this module's functionality.

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

    # Server
    server = FederatedPredictionAggregator(
        dashboard_url=args.dashboard_url,
        window_size=args.window_size,
        name=args.name,
        timeout=args.timeout,
        addr=(args.serv, args.servPort),
    )
    server.start()
    input("Press Enter to stop client...")
    server.stop()


if __name__ == "__main__":
    create_server()
