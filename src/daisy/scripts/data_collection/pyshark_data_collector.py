# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Pre-configured data collection component to collect, pre-process (filter and label),
and write to CSV files, live network traffic data from pyshark (tshark), either from
the local machine or redirected from a remote machine. This component can be
launched directly through python or through the command line option. Note that this
does not launch the remote data source, it has to be run additionally on the remote
host.

Author: Jonathan Ackerschewski
Modified: 25.10.2024
"""

import argparse
import logging
from datetime import datetime

from daisy.communication import StreamEndpoint
from daisy.data_sources import (
    DataHandler,
    DataProcessor,
    LivePysharkDataSource,
    SimpleRemoteDataSource,
    CSVFileRelay,
    DataHandlerRelay,
    EventHandler,
    PysharkProcessor,
)


def parse_event(line: str) -> tuple[datetime, datetime, str, str]:
    """Takes a single line and splits it into start and end time, label, and condition
    for use in the event handler. The line should have the layout:
        start time, end time, label, condition
    The start and end times should be floats representing the time since epoch. The
    label and condition should be strings.

    :param line: A single line
    :return: The start and end times, the label and the condition
    """
    parts = line.split(",")
    start_time = datetime.fromtimestamp(float(parts[0].strip()))
    end_time = datetime.fromtimestamp(float(parts[1].strip()))
    label = parts[2].strip()
    condition = ",".join(parts[3::]).strip()

    return start_time, end_time, label, condition


def _parse_args() -> argparse.Namespace:
    """Creates a parser for the arguments and parses them.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
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

    destination_group = parser.add_argument_group(
        "Data Destination", "These arguments define which destination the data use."
    )
    destination_group = destination_group.add_mutually_exclusive_group(required=True)
    destination_group.add_argument(
        "--to-file",
        action="store_true",
        dest="toFile",
        help="Sends the collected data to a CSV file.",
    )
    destination_group.add_argument(
        "--relay",
        "--remote-destination",
        action="store_false",
        dest="relay",
        help="Sends the collected data to a remotely connected machine.",
    )

    logging_group_main = parser.add_argument_group(
        "Logging", "These arguments define the log level"
    )
    logging_group = logging_group_main.add_mutually_exclusive_group()
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
        help="Increases verbosity with each occurrence up to level 3.",
    )
    logging_group_main.add_argument(
        "--log-file",
        "-lf",
        type=str,
        metavar="FILE",
        help="Writes all log messages to specified file instead of the console.",
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
        help="The port of the local machine, which a remote machine connects to. ("
        "Default: 10980)",
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
        help="Sets a BPF-filter, which will be applied on the captured data. ("
        "Default: )",
    )

    output_file_group = parser.add_argument_group(
        "Output File Configuration", "These arguments configure the output file."
    )
    output_file_group.add_argument(
        "--output-file",
        "-outf",
        type=str,
        metavar="FILE",
        required=True,
        dest="outputFile",
        help="sets the output file location.",
    )
    output_file_exclusive_group = output_file_group.add_mutually_exclusive_group()
    output_file_exclusive_group.add_argument(
        "--csv-header-buffer",
        "--buffer",
        "-b",
        type=int,
        metavar="N",
        default=1000,
        help="Sets the number of packets, which will be used for CSV header "
        "discovery. Higher numbers reduce chance of "
        "missing headers, but increase RAM usage. (Default: 1000)",
    )
    output_file_exclusive_group.add_argument(
        "--headers-file",
        "-hf",
        type=str,
        metavar="FILE",
        help="Sets the headers file location. If this is provided, the auto header "
        "discovery is turned off and the "
        "provided headers will be used instead. Each line is expected to be a "
        "feature/header.",
    )
    output_file_group.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="Set this, to overwrite existing CSV files.",
    )
    output_file_group.add_argument(
        "--separator",
        "-s",
        type=str,
        metavar="CHAR",
        default=",",
        help="Sets the separator for the CSV file. (Default: ,)",
    )
    output_file_group.add_argument(
        "--feature-filter",
        "-ff",
        type=str,
        metavar="FILE",
        help="Removes the features specified in the given file from each packet. The "
        "file is expected to have one "
        "feature per line.",
    )
    output_file_group.add_argument(
        "--events",
        "-e",
        type=str,
        metavar="FILE",
        help="Extracts events for labeling from the given file. Each line is expected "
        "to be one event.",
    )

    output_relay_group = parser.add_argument_group(
        "Output Relay Configuration", "These arguments configure the OUTPUT connection."
    )
    output_relay_group.add_argument(
        "--out-local-ip",
        "-out-ip",
        type=str,
        metavar="IP-ADDRESS",
        help="The IP of the local machine, which a remote machine connects to.",
    )
    output_relay_group.add_argument(
        "--out-local-port",
        "--out-port",
        "-out-p",
        type=int,
        metavar="PORT",
        default=10980,
        help="The port of the local machine, which a remote machine connects to. ("
        "Default: 10980)",
    )
    output_relay_group.add_argument(
        "--out-remote-ip",
        "--out-target-ip",
        "--out-target",
        "-out-tip",
        type=str,
        metavar="IP-ADDRESS",
        help="The IP of the remote machine.",
    )
    output_relay_group.add_argument(
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


def check_args(args):
    """Performs a quick check of some arguments"""

    match args.loglevel:
        case 0:
            log_level = logging.ERROR
        case 1:
            log_level = logging.WARNING
        case 2:
            log_level = logging.INFO
        case _:
            log_level = logging.DEBUG

    if not args.log_file:
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=log_level,
        )
    else:
        logging.basicConfig(
            filename=args.log_file,
            filemode="w",
            format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=log_level,
        )

    if not args.localSource:
        if not args.local_ip or not args.remote_ip:
            logging.error("You must specify a local and remote IP address.")
            exit(1)


def create_data_source(args):
    """Creates and returns the data handler"""

    if args.localSource:
        return LivePysharkDataSource(
            name="data_collector.live_data_source",
            interfaces=args.interfaces,
            bpf_filter=args.filter,
        )
    else:
        stream_endpoint = StreamEndpoint(
            name="data_collector.stream_endpoint",
            addr=(args.local_ip, args.local_port),
            remote_addr=(args.remote_ip, args.remote_port),
            acceptor=True,
            multithreading=args.io_multithreading,
        )
        return SimpleRemoteDataSource(
            endpoint=stream_endpoint,
            name="data_collector.remote_data_source",
        )


def read_collection_files(args):
    """Reads and parses the feature file, headers file and event file and returns the
    three in the order f_features, events, headers.
    """

    if not args.feature_filter:
        f_features = []
    else:
        try:
            with open(args.feature_filter, "r") as feature_file:
                f_features = [line.strip() for line in feature_file]
        except FileNotFoundError:
            logging.error(
                f"Feature filter file does not exist at path {args.feature_filter}."
            )
            exit(1)

    events = EventHandler()
    if args.events:
        try:
            with open(args.events, "r") as events_file:
                for event in events_file:
                    try:
                        start_time, end_time, label, condition = parse_event(event)
                        events.add_event(start_time, end_time, label, condition)
                    except ValueError:
                        logging.warning(
                            f"A line in the event file could not be parsed. Line: "
                            f"{event}"
                        )
                        continue
        except FileNotFoundError:
            logging.error(f"Event file does not exist at path {args.events}.")
            exit(1)
        except Exception as e:
            logging.error("Unexpected error occurred while parsing events file.")
            logging.error(e)
            exit(1)

    headers = None
    if args.headers_file is not None:
        with open(args.headers_file, "r") as headers_file:
            headers = tuple([line.strip() for line in headers_file])

    return f_features, events, headers


def create_data_processor(args, f_features, events):
    """Creates the data processor, which either processes the data points on the
    machine the data is written to file or returns the data points unprocessed
    for relaying to another machine.
    """

    if args.toFile:
        return (
            PysharkProcessor()
            .packet_to_dict()
            .remove_dict_features(f_features)
            .add_func(
                lambda o_point: events.process(
                    datetime.fromtimestamp(float(o_point.get("meta.time_epoch", 0))),
                    o_point,
                )
            )
        )
    else:
        return DataProcessor()


def create_relay(args, data_handler, headers):
    """Creates the relay. This is either a CSVFileRelay if the data should be written
    to file or a DataSourceRelay if the data should be transferred to another machine.
    """

    if args.toFile:
        return CSVFileRelay(
            data_handler=data_handler,
            target_file=args.outputFile,
            name="data_collector.csv_file_relay",
            header_buffer_size=args.csv_header_buffer,
            headers=headers,
            overwrite_file=args.overwrite,
            separator=args.separator,
            default_missing_value="",
        )
    else:
        stream_endpoint_out = StreamEndpoint(
            name="data_relay.stream_endpoint_out",
            addr=(args.out_local_ip, args.out_local_port),
            remote_addr=(args.out_remote_ip, args.out_remote_port),
            acceptor=False,
        )
        return DataHandlerRelay(
            data_handler=data_handler,
            endpoint=stream_endpoint_out,
            name="data_relay.data_relay",
        )


def create_collector():
    """Creates a CSV file relay with all needed structures to provide it with data.
    There is the option to either use a live data capture on the local machine or to
    use data from a remote machine.
    """
    # Args parsing
    args = _parse_args()
    check_args(args)

    data_source = create_data_source(args)
    f_features, events, headers = read_collection_files(args)
    data_processor = create_data_processor(args, f_features, events)

    data_handler = DataHandler(
        data_source=data_source,
        data_processor=data_processor,
        name="data_collector.data_handler",
        multithreading=args.processing_multithreading,
    )

    relay = create_relay(args, data_handler, headers)

    relay.start()
    input("Press Enter to stop server...")
    relay.stop()


if __name__ == "__main__":
    create_collector()
