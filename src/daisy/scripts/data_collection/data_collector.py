# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""TODO

Author: Jonathan Ackerschewski
Modified: TODO
"""

import argparse
import logging


def _parse_args() -> argparse.Namespace:
    """Creates a parser for the server arguments and parses them.

    :return: Parsed arguments.
    """
    # TODO add arguments
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--debug", type=bool, default=False, metavar="", help="Show debug outputs"
    )
    parser.add_argument(
        "--serv", required=True, help="IP or hostname of evaluation aggregation server"
    )
    parser.add_argument(
        "--servPort",
        type=int,
        default=8001,
        choices=range(1, 65535),
        metavar="",
        help="Port of evaluation aggregation server",
    )

    aggr_options = parser.add_argument_group("Aggregator Options")
    aggr_options.add_argument(
        "--timeout",
        type=int,
        default=10,
        metavar="",
        help="Timeout to receive metric updates from federated nodes",
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


def create_collector():
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

    # TODO create the collector


if __name__ == "__main__":
    create_collector()
