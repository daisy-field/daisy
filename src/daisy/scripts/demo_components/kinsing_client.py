# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

Author: Fabian Hofmann
Modified: 04.11.24
"""

import argparse
import logging
import pathlib

import keras
import numpy as np

from daisy.data_sources import (
    DataHandler,
    pcap_nn_aggregator,
)
from daisy.evaluation import ConfMatrSlidingWindowEvaluation
from daisy.federated_ids_components import FederatedOnlineClient
from daisy.federated_learning import TFFederatedModel

from daisy.data_sources import CSVFileDataSource
from daisy.federated_learning.threshold_models import kinsingTM

from daisy.data_sources import DataProcessor


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
        "--csvBasePath",
        type=pathlib.Path,
        default="/home/fabian/Documents/DAI-Lab/COBRA-5G/D-IDS/Datasets/v2x_2023-03-06",
        metavar="",
        help="Path to the kinsing dataset directory (root)",
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

    # Datasource
    source = CSVFileDataSource(f"{args.csvBasePath}/labelled_kinsing.csv")
    kinsing_f_features = (
        "data.len",
        "eth.addr",
        "eth.dst",
        "eth.len",
        "eth.src",
        "ip.addr",
        "ip.dst",
        "ip.flags.df",
        "ip.flags.mf",
        "ip.flags.rb",
        "ip.src",
        "ipv6.addr",
        "ipv6.dst",
        "ipv6.src",
        "ipv6.tclass",
        "llc.control.ftype",
        "llc.control.n_r",
        "llc.control.n_s",
        "llc.dsap.ig",
        "llc.dsap.sap",
        "llc.ssap.cr",
        "llc.ssap.sap",
        "meta.len",
        "meta.number",
        "meta.protocols",
        "meta.time",
        "sll.eth",
        "sll.etype",
        "sll.halen",
        "sll.hatype",
        "sll.ltype",
        "sll.padding",
        "sll.pkttype",
        "sll.trailer",
        "sll.unused",
        "ssh.direction",
        "ssh.protocol",
        "tcp.dstport",
        "tcp.port",
        "tcp.segments.count",
        "tcp.srcport",
        "udp.port",
        "udp.payload",
        "tcp.payload",
    )
    processor = (  # tcp.payload and udp.payload contain the payload. Calculate length of both and add them to the last row of
        DataProcessor()
        .select_dict_features(features=kinsing_f_features, default_value=np.nan)
        .cast_dict_features(["ip.addr"], [str])
        .shrink_payload()
        .dict_to_array(nn_aggregator=pcap_nn_aggregator)
    )
    data_handler = DataHandler(data_source=source, data_processor=processor)

    # Model
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss = keras.losses.MeanAbsoluteError()
    id_fn = TFFederatedModel.get_fae(
        input_size=44,
        optimizer=optimizer,
        loss=loss,
        batch_size=args.batchSize,
        epochs=1,
    )
    t_m = kinsingTM()
    err_fn = keras.losses.MeanAbsoluteError(reduction=None)
    model = TFFederatedModel(identify_fn=id_fn, threshold_m=t_m, error_fn=err_fn)

    metrics = [ConfMatrSlidingWindowEvaluation(window_size=args.batchSize * 8)]

    # Client
    client = FederatedOnlineClient(
        data_handler=data_handler,
        batch_size=args.batchSize,
        model=model,
        label_split=44,
        metrics=metrics,
        m_aggr_server=m_aggr_serv,
        eval_server=eval_serv,
        aggr_server=aggr_serv,
        update_interval_t=args.updateInterval,
    )
    client.start()
    input("Press Enter to stop client...")
    client.stop()


if __name__ == "__main__":
    create_client()
