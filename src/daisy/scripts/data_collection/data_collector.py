# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""This script runs a data collector, which can either capture live data from the local machine or gather data from a remote connection. The data will be written into CSV files.

Author: Jonathan Ackerschewski
Modified: 03.07.2024
"""

import argparse
import logging

from daisy.data_sources import (
    DataSource,
    SimpleDataProcessor,
    pyshark_map_fn,
    LivePysharkHandler,
    SimpleRemoteSourceHandler,
    CSVFileRelay,
)
from daisy.communication import StreamEndpoint


def _parse_args() -> argparse.Namespace:
    """Creates a parser for the arguments and parses them.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    source_group = parser.add_argument_group(
        "Data Source", "These arguments define which source for the data to use."
    )
    source_group = source_group.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--local",
        "--local-source",
        action="store_true",
        dest="localSource",
        help="Collects data from the local machine.",
    )
    source_group.add_argument(
        "--remote",
        "--remote-source",
        action="store_false",
        dest="localSource",
        help="Collects data from a remotely connected machine.",
    )

    logging_group = parser.add_argument_group(
        "Logging", "These arguments define the log level"
    )
    logging_group = logging_group.add_mutually_exclusive_group()
    logging_group.add_argument(
        "--debug",
        action="store_const",
        const=3,
        default=0,
        dest="loglevel",
        help="Sets the logging level to debug (level 3). Equivalent to -vvv.",
    )
    logging_group.add_argument(
        "--info",
        action="store_const",
        const=2,
        default=0,
        dest="loglevel",
        help="Sets the logging level to info (level 2). Equivalent to -vv.",
    )
    logging_group.add_argument(
        "--warning",
        "--warn",
        action="store_const",
        const=1,
        default=0,
        dest="loglevel",
        help="Sets the logging level to warning (level 1). Equivalent to -v.",
    )
    logging_group.add_argument(
        "--v",
        "--verbose",
        "-v",
        action="count",
        default=0,
        dest="loglevel",
        help="Increases verbosity with each occurance up to level 3.",
    )

    remote_group = parser.add_argument_group(
        "Remote Source", "These arguments are required for use with remote sources."
    )
    remote_group.add_argument(
        "--local-ip",
        "-ip",
        type=str,
        metavar="IP-ADDRESS",
        help="The IP of the local machine, which a remote machine connects to.",
    )
    remote_group.add_argument(
        "--local-port",
        "--port",
        "-p",
        type=int,
        metavar="PORT",
        default=10980,
        help="The port of the local machine, which a remote machine connects to. (Default: 10980)",
    )
    remote_group.add_argument(
        "--remote-ip",
        "--target-ip",
        "--target",
        "-tip",
        type=str,
        metavar="IP-ADDRESS",
        help="The IP of the remote machine.",
    )
    remote_group.add_argument(
        "--remote-port",
        "--target-port",
        "-tp",
        type=int,
        metavar="PORT",
        default=10980,
        help="The port of the remote machine. (Default: 10980)",
    )

    local_group = parser.add_argument_group(
        "Local Source", "These arguments are required for use with local sources."
    )
    local_group.add_argument(
        "--interfaces",
        "-i",
        type=str,
        metavar="INTERFACES",
        default=["any"],
        nargs="*",
        help="Sets the interfaces, which should be captured. (Default: any)",
    )
    local_group.add_argument(
        "--filter",
        "-f",
        type=str,
        metavar="BPF-FILTER",
        default="",
        help="Sets a BPF-filter, which will be applied on the captured data. (Default: )",
    )

    output_group = parser.add_argument_group(
        "Output Configuration", "These arguments configure the output."
    )
    output_group.add_argument(
        "--output",
        "-out",
        type=str,
        metavar="FILE",
        required=True,
        help="sets the output file location.",
    )
    output_group.add_argument(
        "--csv-header-buffer",
        "--buffer",
        "-b",
        type=int,
        metavar="N",
        default=1000,
        help="Sets the number of packets, which will be used for CSV header discovery. Higher numbers reduce chance of missing headers, but increase RAM usage. (Default: 1000)",
    )
    output_group.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="Set this, to overwrite existing CSV files.",
    )
    output_group.add_argument(
        "--separator",
        "-s",
        type=str,
        metavar="CHAR",
        default=",",
        help="Sets the separator for the CSV file. (Default: ,)",
    )

    performance_group = parser.add_argument_group(
        "Performance Configuration", "These arguments can adjust performance."
    )
    performance_group.add_argument(
        "--io-multithreading",
        "-io-mt",
        action="store_true",
        help="Enables multi-threading for IO operations. Only relavant for remote data sources.",
    )
    performance_group.add_argument(
        "--processing-multithreading",
        "-p-mt",
        action="store_true",
        help="Enables multi-threading for packet processing.",
    )

    return parser.parse_args()


def create_collector():
    """Creates a CSV file relay with all needed structures to provide it with data. There is the option to either use a live data capture on the local machine or to use data from a remote machine."""
    # Args parsing
    args = _parse_args()

    match args.loglevel:
        case 0:
            log_level = logging.ERROR
        case 1:
            log_level = logging.WARNING
        case 2:
            log_level = logging.INFO
        case _:
            log_level = logging.DEBUG

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=log_level,
    )

    if not args.localSource:
        if not args.local_ip or not args.remote_ip:
            logging.error("You must specify a local and remote IP address.")
            exit(1)

    if args.localSource:
        data_handler = LivePysharkHandler(
            name="data_collector:live_data_handler",
            interfaces=args.interfaces,
            bpf_filter=args.filter,
        )
    else:
        stream_endpoint = StreamEndpoint(
            name="data_collector:stream_endpoint",
            addr=(args.local_ip, args.local_port),
            remote_addr=(args.remote_ip, args.remote_port),
            acceptor=True,
            multithreading=args.io_multithreading,
        )
        data_handler = SimpleRemoteSourceHandler(
            endpoint=stream_endpoint, name="data_collector:remote_data_handler"
        )
    data_processor = SimpleDataProcessor(
        map_fn=pyshark_map_fn(), filter_fn=lambda x: x, reduce_fn=lambda x: x
    )
    data_source = DataSource(
        source_handler=data_handler,
        data_processor=data_processor,
        name="data_collector:data_source",
        multithreading=args.processing_multithreading,
    )
    csv_file_relay = CSVFileRelay(
        data_source=data_source,
        target_file=args.output,
        name="data_collector:csv_file_relay",
        header_buffer_size=args.csv_header_buffer,
        overwrite_file=args.overwrite,
        separator=args.separator,
        default_missing_value="",
    )

    csv_file_relay.start()
    input("Press Enter to stop server...")
    csv_file_relay.stop()


if __name__ == "__main__":
    create_collector()
