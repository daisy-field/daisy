# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Pre-configured prediction aggregation server for any federated system, whose nodes
report their predictions to a central server. Making this component auxiliary in
nature, as it merely serves as a central collection agent for further processing or
visualization purposes.

Note this server does nothing on its own --- it requires at least one active
federated node to be run in tandem with it, whose set to report its prediction values
to this server.

Author: Fabian Hofmann, Seraphin Zunzer
Modified: 17.06.24
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
        default=8002,
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
        help="Timeout to receive prediction updates from federated nodes",
    )
    aggr_options.add_argument(
        "--windowSize",
        type=int,
        default=None,
        metavar="",
        help="Window size of aggregator",
    )
    aggr_options.add_argument(
        "--dashboardURL", default="127.0.0.1", help="IP of (external) dashboard server"
    )

    return parser.parse_args()


def create_server():
    """Creates a pre-configured prediction value collection server node.
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

    # Server
    server = FederatedPredictionAggregator(
        addr=(args.serv, args.servPort),
        window_size=args.windowSize,
        timeout=args.timeout,
        dashboard_url=args.dashboardURL,
    )
    server.start()
    input("Press Enter to stop server...")
    server.stop()


if __name__ == "__main__":
    create_server()
