import sys

import argparse
import logging
import pathlib

from daisy.data_sources import DataSource
from daisy.data_sources import DataHandler,DataProcessor, march23_events

import pandas as pd


def _parse_args() -> argparse.Namespace:
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
        choices=[2, 5],
        required=True,
        help="ID of client (decides which data to draw from set)",
    )
    parser.add_argument(
        "--pcapBasePath",
        type=pathlib.Path,
        default="/mnt/h/daisy_datasets/v2x_2023-03-06",
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

sys.argv = ["script.py", "--clientId=2"]

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
m_aggr_serv = (args.modelAggrServ, args.modelAggrServPort)
eval_serv = None
if args.evalServ != "0.0.0.0":
    eval_serv = (args.evalServ, args.evalServPort)
aggr_serv = None
if args.aggrServ != "0.0.0.0":
    aggr_serv = (args.aggrServ, args.aggrServPort)

import subprocess

# Der Befehl, den du ausführen möchtest
command = "ls -a -l /home/"

# Die Ausgabe des Befehls abrufen
output = subprocess.check_output(command, shell=True)

# Die Ausgabe in eine Zeichenkette umwandeln und anzeigen
print(output.decode("utf-8"))

import os

# Pfad zum Ordner, den du überprüfen möchtest
ordner_pfad = '/home/daisy_datasets/v2x_2023-03-06'

# Überprüfe, ob der Ordner existiert
if os.path.exists(ordner_pfad):
    print("Der Ordner existiert.")

    # Zeige alle Unterverzeichnisse an
    unterverzeichnisse = [verzeichnis for verzeichnis in os.listdir(ordner_pfad) if os.path.isdir(os.path.join(ordner_pfad, verzeichnis))]
    print("Unterverzeichnisse:")
    for verzeichnis in unterverzeichnisse:
        print(verzeichnis)
else:
    print("Der Ordner existiert nicht.")

logging.info(f"/home/daisy_datasets/v2x_2023-03-06/diginet-cohda-box-dsrc{args.clientId}")

import os
cwd = os.getcwd()
cw2 = os.path.abspath("")

logging.info(cwd)
logging.info(cw2)

#handler = PcapHandler(f"{args.pcapBasePath}/diginet-cohda-box-dsrc{args.clientId}/diginet-cohda-box-dsrc2-capture.pcap")
handler = DataHandler(f"/home/daisy_datasets/v2x_2023-03-06/diginet-cohda-box-dsrc{args.clientId}")

processor = DataProcessor(client_id=args.clientId, events=march23_events, reduce_d_point=False)
data_source = DataSource(source_handler=handler, data_processor=processor)

data_source.open()

import os
os.environ["PATH"] += os.pathsep + "/mnt/c/Program Files/Wireshark/tshark.exe"
print(os.environ['PATH'])


output = pd.DataFrame()

def list_to_tuple(x):
    if isinstance(x, list):
        return tuple(x)
    return x

output1= output.applymap(list_to_tuple)

output1.duplicated(keep=False).any()

for sample in data_source:
    df_dictionary = pd.DataFrame([sample])
    output = pd.concat([output, df_dictionary], ignore_index=True)

    # check duplicats
    output1 = output.applymap(list_to_tuple)
    duplicate = output1.duplicated(keep=False).any()

    if duplicate:
        logging.info("duplicate detected")
        break

output.to_csv("Test.csv", encoding='utf-8')

logging.info("Ende")
