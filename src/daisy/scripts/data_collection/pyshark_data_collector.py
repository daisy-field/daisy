# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""This script runs a data collector, which can either capture live data from the
local machine or gather data from a remote connection. The data will be written into
CSV files.

Author: Jonathan Ackerschewski
Modified: 03.07.2024
"""

import argparse
import logging
import re

from daisy.data_sources import (
    DataSource,
    SimpleDataProcessor,
    IdentityDataProcessor,
    remove_filter_fn,
    pyshark_map_fn,
    LivePysharkHandler,
    SimpleRemoteSourceHandler,
    CSVFileRelay,
    DataSourceRelay,
)
from daisy.communication import StreamEndpoint


def label_reduce(
    d_point: dict,
    events: list[
        tuple[tuple[float, float], list[tuple[any, any]], list[tuple[any, any]], str]
    ],
    default_label: str = "",
) -> dict:
    d_point["label"] = default_label
    for event in events:
        if not (event[0][0] < float(d_point.get("meta.time_epoch", 0)) < event[0][1]):
            continue

        skip = False
        for feature, value in event[1]:
            if d_point.get(feature, None) != value:
                skip = True
                break
        if skip:
            continue

        skip = False
        for feature, value in event[2]:
            try:
                if value not in d_point.get(feature, []):
                    skip = True
                    break
            except TypeError:
                val = d_point.get(feature, [])
                if val:
                    logging.error(
                        f"Got TypeError for feature {feature}, trying to determine if "
                        f"{value} is in "
                        f"{val}. Skipping..."
                    )
                    skip = True
                    break
        if skip:
            continue

        d_point["label"] = event[3]
    return d_point


def parse_event(line):
    parts = re.split("[\[\]]", line)
    times = parts[0].split(",")
    times[0] = float(times[0])
    times[1] = float(times[1])

    eq_parts = re.split("[()]", parts[1])
    eq_list = []
    for eq in eq_parts:
        cur_eq = [
            keyval.strip().strip('"')
            for keyval in re.split(',(?=(?:[^"]*"[^"]*")*[^"]*$)', eq)
        ]
        if len(cur_eq) != 2:
            continue
        key, val = cur_eq
        if not key and not val:
            continue
        eq_list += [(key, val)]

    con_parts = re.split("[()]", parts[3])
    con_list = []
    for con in con_parts:
        cur_con = [
            keyval.strip().strip('"')
            for keyval in re.split(',(?=(?:[^"]*"[^"]*")*[^"]*$)', con)
        ]
        if len(cur_con) != 2:
            continue
        key, val = cur_con
        if not key and not val:
            continue
        con_list += [(key, val)]

    label = parts[4].strip(",").strip().strip('"')

    return ((times[0], times[1]), eq_list, con_list, label)


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
        required=True,
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
        required=True,
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


def create_data_handler(args):
    """Creates and returns the data handler"""

    if args.localSource:
        return LivePysharkHandler(
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
        return SimpleRemoteSourceHandler(
            endpoint=stream_endpoint, name="data_collector:remote_data_handler"
        )


def read_collection_files(args):
    """Reads and parses the feature file, headers file and event file and returns the three in the order
    f_features, events, headers."""

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

    if not args.events:
        events = []
    else:
        try:
            with open(args.events, "r") as events_file:
                events = []
                for event in events_file:
                    try:
                        events += [parse_event(event)]
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
        logging.info("Found following events:")
        for event in events:
            logging.info(
                f"Start time: {event[0][0]}, End time: {event[0][1]}, Label: "
                f"{event[3]}, List of feature equals value: "
                f"{event[1]}, List of value in feature: {event[2]}"
            )

    headers = None
    if args.headers_file is not None:
        with open(args.headers_file, "r") as headers_file:
            headers = tuple([line.strip() for line in headers_file])

    return f_features, events, headers


def create_data_processor(args, f_features, events):
    """Creates the data processor, which either processes the data points on the machine the data is written to file
    or returns the data points unprocessed for relaying to another machine."""

    if args.toFile:
        return SimpleDataProcessor(
            map_fn=pyshark_map_fn(),
            filter_fn=remove_filter_fn(f_features),
            reduce_fn=lambda x: label_reduce(x, events, default_label="benign"),
        )
    else:
        return IdentityDataProcessor()


def create_relay(args, data_source, headers):
    """Creates the relay. This is either a CSVFileRelay if the data should be written to file or a DataSourceRelay
    if the data should be transferred to another machine."""

    if args.toFile:
        return CSVFileRelay(
            data_source=data_source,
            target_file=args.outputFile,
            name="data_collector:csv_file_relay",
            header_buffer_size=args.csv_header_buffer,
            headers=headers,
            overwrite_file=args.overwrite,
            separator=args.separator,
            default_missing_value="",
        )
    else:
        stream_endpoint_out = StreamEndpoint(
            name="data_relay:stream_endpoint_out",
            addr=(args.out_local_ip, args.out_local_port),
            remote_addr=(args.out_remote_ip, args.out_remote_port),
            acceptor=False,
        )
        return DataSourceRelay(
            data_source=data_source,
            endpoint=stream_endpoint_out,
            name="data_relay:data_relay",
        )


def create_collector():
    """Creates a CSV file relay with all needed structures to provide it with data.
    There is the option to either use
    a live data capture on the local machine or to use data from a remote machine."""
    # Args parsing
    args = _parse_args()
    check_args(args)

    data_handler = create_data_handler(args)
    f_features, events, headers = read_collection_files(args)
    data_processor = create_data_processor(args, f_features, events)

    data_source = DataSource(
        source_handler=data_handler,
        data_processor=data_processor,
        name="data_collector:data_source",
        multithreading=args.processing_multithreading,
    )

    relay = create_relay(args, data_source, headers)

    relay.start()
    input("Press Enter to stop server...")
    relay.stop()


if __name__ == "__main__":
    create_collector()
