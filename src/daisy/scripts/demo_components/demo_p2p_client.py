# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""First draft of a p2p client.

Author: Seraphin Zunzer, Fabian Hofmann, Lotta Fejzula
Modified: 02.03.25
"""
# TODO bring in the same format as other demo clients

import argparse
import logging
import pathlib

import numpy as np
import tensorflow as tf

from daisy.data_sources import (
    DataHandler,
    PcapDataSource,
    pcap_nn_aggregator,
    PysharkProcessor,
)

from daisy.demos.v2x_23_03 import pcap_f_features, demo_202303_label_data_point


from daisy.evaluation import ConfMatrSlidingWindowEvaluation
from daisy.federated_learning import (
    FederatedIFTM,
    FedAvgAggregator,
    TFFederatedModel,
    MadTM,
)

from daisy.federated_ids_components import FederatedOnlinePeer


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--port", type=int, default=None, help="Port of peer")
    parser.add_argument(
        "--joinPort", type=int, default=None, help="Port of peer to join dht on"
    )
    parser.add_argument(
        "--evalServ",
        default="localhost",
        metavar="",
        help="IP or hostname of evaluation server",
    )
    parser.add_argument(
        "--evalServPort",
        type=int,
        default=8001,
        choices=range(1, 65535),
        metavar="",
        help="Port of evaluation server",
    )
    parser.add_argument(
        "--pcapBasePath",
        type=pathlib.Path,
        default="/home/fabian/Repositories/datasets/v2x_2023-03-06",
        metavar="",
        help="Path to the march23 v2x dataset directory (root)",
    )
    parser.add_argument(
        "--clientId",
        type=int,
        choices=[2, 5],
        required=True,
        help="ID of client (decides which data to draw from set)",
    )
    parser.add_argument(
        "--batchSize",
        type=int,
        default=256,
        metavar="",
        help="Batch size during processing of data "
        "(mini-batches are multiples of that argument)",
    )
    args = parser.parse_args()

    eval_serv = None
    if args.evalServ != "0.0.0.0":
        eval_serv = (args.evalServ, args.evalServPort)

    # Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss = tf.keras.losses.MeanAbsoluteError()
    id_fn = TFFederatedModel.get_fae(
        input_size=65,
        optimizer=optimizer,
        loss=loss,
        batch_size=args.batchSize,
        epochs=1,
    )

    t_m = MadTM(window_size=args.batchSize * 8, threshold=2.2)
    err_fn = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    model = FederatedIFTM(identify_fn=id_fn, threshold_m=t_m, error_fn=err_fn)

    metrics = [ConfMatrSlidingWindowEvaluation(window_size=args.batchSize * 8)]

    source = PcapDataSource(
        f"{args.pcapBasePath}/diginet-cohda-box-dsrc{args.clientId}"
    )
    processor = (
        PysharkProcessor()
        .packet_to_dict()
        .select_dict_features(features=pcap_f_features, default_value=np.nan)
        .add_func(
            lambda o_point: demo_202303_label_data_point(
                client_id=args.clientId, d_point=o_point
            )
        )
        .dict_to_array(nn_aggregator=pcap_nn_aggregator)
    )
    data_handler = DataHandler(data_source=source, data_processor=processor)

    join_addr = None
    if args.joinPort:
        join_addr = ("127.0.0.1", args.joinPort)
    federated_node = FederatedOnlinePeer(
        data_handler=data_handler,
        model=model,
        batch_size=args.batchSize * 8,
        m_aggr=FedAvgAggregator(),
        eval_server=eval_serv,
        port=int(args.port),
        metrics=metrics,
        dht_join_addr=join_addr,
        label_split=65,
    )
    federated_node.start()
    input("Press Enter to stop peer...")
    federated_node.stop()
