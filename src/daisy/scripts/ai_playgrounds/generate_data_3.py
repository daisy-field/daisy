import argparse
import logging
import pathlib

from daisy.data_sources import DataSource
from daisy.data_sources import PcapHandler, CohdaProcessor, march23_events
from daisy.data_sources import CSVFileRelay

import pandas as pd

def _parse_args(clientId: int) -> argparse.Namespace:
    """Creates a parser for the client arguments and parses them.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--debug", type=bool, default=False, metavar="", help="Show debug outputs"
    )
    parser.add_argument(
        "--clientId",
        type=int,
        default=clientId,
        choices=[2, 5],
        required=True,
        help="ID of client (decides which data to draw from set)",
    )
    parser.add_argument(
        "--pcapBasePath",
        type=pathlib.Path,
        default=f"/home/daisy_datasets/v2x_2023-03-06/diginet-cohda-box-dsrc{clientId}",
        metavar="",
        help="Path to the march23 v2x dataset directory (root)",
    )

    server_options = parser.add_argument_group("Server Options")
    server_options.add_argument(
        "--modelAggrServ",
        default="0.0.0.0",
        metavar="",
        help="IP or hostname of model aggregation server",
    )
    server_options.add_argument(
        "--modelAggrServPort",
        type=int,
        default=8000,
        choices=range(1, 65535),
        metavar="",
        help="Port of model aggregation server",
    )
    server_options.add_argument(
        "--evalServ",
        default="0.0.0.0",
        metavar="",
        help="IP or hostname of evaluation server",
    )
    server_options.add_argument(
        "--evalServPort",
        type=int,
        default=8001,
        choices=range(1, 65535),
        metavar="",
        help="Port of evaluation server",
    )
    server_options.add_argument(
        "--aggrServ",
        default="0.0.0.0",
        metavar="",
        help="IP or hostname of aggregation server",
    )
    server_options.add_argument(
        "--aggrServPort",
        type=int,
        default=8002,
        choices=range(1, 65535),
        metavar="",
        help="Port of aggregation server",
    )

    client_options = parser.add_argument_group("Client Options")
    client_options.add_argument(
        "--batchSize",
        type=int,
        default=32,
        metavar="",
        help="Batch size during processing of data "
        "(mini-batches are multiples of that argument)",
    )
    client_options.add_argument(
        "--updateInterval",
        type=int,
        default=None,
        metavar="",
        help="Federated updating interval, defined by time (s)",
    )

    return parser.parse_args()


def create_client():
    """Creates a pre-configured federated client with preset components that runs on
    either of the two subsets of the March 6th 2023 network traffic data set. Entry
    point of this module's functionality.

    See the header doc string of this module for more details about the preset
    client's configuration.
    """
    # Args parsing
    args = _parse_args(clientId = 2)
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

    # Datasource
    handler = PcapHandler(f"{args.pcapBasePath}/diginet-cohda-box-dsrc{args.clientId}")
    processor = CohdaProcessor(client_id=args.clientId, events=march23_events)
    data_source = DataSource(source_handler=handler, data_processor=processor)

    csv_file_name = f"csv-example-full-dsrc_{args.clientId}.csv"

    csv = CSVFileRelay(data_source=data_source,
                       target_file=csv_file_name,  # Output Datei
                       overwrite_file=True,  # Bool, ob der Output überschrieben werden soll (praktisch zum testen)
                       separator=",",  # der separator in der csv
                       default_missing_value="")  # Welcher Value wird verwendet, wenn die Datenpunkt keinen Wert in der Spalte hat

    csv.start()
    csv.stop()

    df = pd.read_csv(csv_file_name)

    labels = df['label'].unique()


if __name__ == "__main__":
    create_client()
