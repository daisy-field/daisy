# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Pre-configured demonstration client for a simple federated intrusion detection
system (IDS), that learns cooperatively with another clients through a centralized
model aggregation server using the federated averaging (FedAvg) technique. In this
example, the client is configured to process network traffic data from the CICIDS17
dataset.

There are also different options to choose pFL at startup.

Author: Fabian Hofmann, Seraphin Zunzer
Modified: 13.01.25
"""

import argparse
import logging
import pathlib

import tensorflow as tf

from daisy.data_sources import (
    DataHandler,
)
from daisy.evaluation import ConfMatrSlidingWindowEvaluation
from daisy.data_sources import CSVFileDataSource, DataProcessor

from daisy.data_sources.network_traffic.cic import (
    cic_label_data_point,
    csv_nn_aggregator,
)

from daisy.federated_learning import SMAvgTM

from daisy.scripts.demo_components.fl_configs import (
    load_generative_pfl_conf,
    load_distillative_pfl_conf,
    load_traditional_fl_conf,
)


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
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        required=True,
        help="ID of client (decides which data to draw from set)",
    )
    parser.add_argument(
        "--csvBasePath",
        type=pathlib.Path,
        default="/home/fabian/Documents/DAI-Lab/COBRA-5G/D-IDS/Datasets/v2x_2023-03-06",
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
        default=8003,
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
    client_options.add_argument(
        "--pflMode",
        type=str,
        choices=["generative", "distillative", "layerwise"],
        default=None,
        metavar="",
        help="Enable personalized FL and choose desired method: generative, distillative, layerwise",
    )
    client_options.add_argument(
        "--autoModel",
        type=bool,
        default=False,
        metavar="",
        help="Select local model size in personalized FL automatically based on hardware constraints. ",
    )
    client_options.add_argument(
        "--manualModel",
        type=str,
        default=None,
        choices=["small", "medium", "large"],
        metavar="",
        help="Select local model size in personalized FL manually.",
    )
    client_options.add_argument(
        "--poisoningMode",
        type=str,
        default=None,
        choices=["zeros", "random", "inverse"],
        metavar="",
        help="Configure the model poisoning mode.",
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

    source = CSVFileDataSource(files=f"{args.csvBasePath}/{args.clientId}.csv")
    processor = (
        DataProcessor()
        .add_func(lambda o_point: cic_label_data_point(d_point=o_point))
        .dict_to_array(nn_aggregator=csv_nn_aggregator)
    )

    data_handler = DataHandler(data_source=source, data_processor=processor)
    t_m = SMAvgTM()
    err_fn = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

    metrics = [ConfMatrSlidingWindowEvaluation(window_size=args.batchSize * 8)]

    if args.pflMode is None:
        load_traditional_fl_conf(
            args,
            t_m,
            err_fn,
            data_handler,
            metrics,
            m_aggr_serv,
            eval_serv,
            aggr_serv,
            input_size=78,
            label_split=78,
        )

    if args.pflMode == "generative":
        load_generative_pfl_conf(
            args,
            t_m,
            err_fn,
            data_handler,
            metrics,
            m_aggr_serv,
            eval_serv,
            aggr_serv,
            input_size=78,
            label_split=78,
        )

    if args.pflMode == "distillative":
        load_distillative_pfl_conf(
            args,
            t_m,
            err_fn,
            data_handler,
            metrics,
            m_aggr_serv,
            eval_serv,
            aggr_serv,
            input_size=78,
            label_split=78,
        )

    if args.pflMode == "layerwise":
        raise NotImplementedError


if __name__ == "__main__":
    create_client()
