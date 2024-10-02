# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""This script starts a relay, which passes data to a remote machine for further
processing. There it the option to use
locally captured data or use data provided by a remote source.

Author: Jonathan Ackerschewski
Modified: 03.07.2024
"""

import argparse
import logging

from daisy.data_sources import (
    DataSource,
    SimpleDataProcessor,
    LivePysharkHandler,
    SimpleRemoteSourceHandler,
    DataSourceRelay,
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
        "Remote Source",
        "These arguments are required for use with remote sources. They specify the "
        "data INPUT connection.",
    )
    remote_group.add_argument(
        "--in-local-ip",
        "-in-ip",
        type=str,
        metavar="IP-ADDRESS",
        help="The IP of the local machine, which a remote machine connects to.",
    )
    remote_group.add_argument(
        "--in-local-port",
        "--in-port",
        "-in-p",
        type=int,
        metavar="PORT",
        default=10980,
        help="The port of the local machine, which a remote machine connects to. ("
        "Default: 10980)",
    )
    remote_group.add_argument(
        "--in-remote-ip",
        "--in-target-ip",
        "--in-target",
        "-in-tip",
        type=str,
        metavar="IP-ADDRESS",
        help="The IP of the remote machine.",
    )
    remote_group.add_argument(
        "--in-remote-port",
        "--in-target-port",
        "-in-tp",
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
        help="Sets a BPF-filter, which will be applied on the captured data. ("
        "Default: )",
    )

    output_group = parser.add_argument_group(
        "Output Configuration", "These arguments configure the OUTPUT connection."
    )
    output_group.add_argument(
        "--out-local-ip",
        "-out-ip",
        type=str,
        metavar="IP-ADDRESS",
        required=True,
        help="The IP of the local machine, which a remote machine connects to.",
    )
    output_group.add_argument(
        "--out-local-port",
        "--out-port",
        "-out-p",
        type=int,
        metavar="PORT",
        default=10980,
        help="The port of the local machine, which a remote machine connects to. ("
        "Default: 10980)",
    )
    output_group.add_argument(
        "--out-remote-ip",
        "--out-target-ip",
        "--out-target",
        "-out-tip",
        type=str,
        metavar="IP-ADDRESS",
        required=True,
        help="The IP of the remote machine.",
    )
    output_group.add_argument(
        "--out-remote-port",
        "--out-target-port",
        "-out-tp",
        type=int,
        metavar="PORT",
        default=10980,
        help="The port of the remote machine. (Default: 10980)",
    )

    performance_group = parser.add_argument_group(
        "Performance Configuration", "These arguments can adjust performance."
    )
    performance_group.add_argument(
        "--io-multithreading",
        "-io-mt",
        action="store_true",
        help="Enables multi-threading for IO operations. Only relavant for remote "
        "data sources.",
    )
    performance_group.add_argument(
        "--processing-multithreading",
        "-p-mt",
        action="store_true",
        help="Enables multi-threading for packet processing.",
    )

    return parser.parse_args()


def create_relay():
    """Creates a relay, which passes provided data to a remote machine for further
    processing. There it the option to
    use locally captured data or use data provided by a remote source."""
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
        if not args.in_local_ip or not args.in_remote_ip:
            logging.error("You must specify a local and remote IP address.")
            exit(1)

    if args.localSource:
        data_handler = LivePysharkHandler(
            name="data_relay:live_data_handler",
            interfaces=args.interfaces,
            bpf_filter=args.filter,
        )
    else:
        stream_endpoint = StreamEndpoint(
            name="data_relay:stream_endpoint",
            addr=(args.local_ip, args.local_port),
            remote_addr=(args.remote_ip, args.remote_port),
            acceptor=True,
            multithreading=args.io_multithreading,
        )
        data_handler = SimpleRemoteSourceHandler(
            endpoint=stream_endpoint, name="data_relay:remote_data_handler"
        )
    data_processor = SimpleDataProcessor(
        map_fn=lambda x: x, filter_fn=lambda x: x, reduce_fn=lambda x: x
    )
    data_source = DataSource(
        source_handler=data_handler,
        data_processor=data_processor,
        name="data_relay:data_source",
        multithreading=args.processing_multithreading,
    )

    stream_endpoint_out = StreamEndpoint(
        name="data_relay:stream_endpoint_out",
        addr=(args.out_local_ip, args.out_local_port),
        remote_addr=(args.out_remote_ip, args.out_remote_port),
        acceptor=False,
    )
    data_relay = DataSourceRelay(
        data_source=data_source,
        endpoint=stream_endpoint_out,
        name="data_relay:data_relay",
    )

    data_relay.start()
    input("Press Enter to stop server...")
    data_relay.stop()


if __name__ == "__main__":
    create_relay()
