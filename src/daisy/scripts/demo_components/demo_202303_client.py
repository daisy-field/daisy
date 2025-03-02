# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Pre-configured demonstration client for a simple federated intrusion detection
system (IDS), that learns cooperatively with another clients through a centralized
model aggregation server using the federated averaging (FedAvg) technique. In this
example, the client is configured to process network traffic data from the road-side
infrastructure (BeIntelli) on Cohda boxes 2 and 5 on March 6th 2023, which must be
available in (raw) pcap files for each client.

The processing is done in online manner (as is the general nature of all current
federated processing nodes), with the underlying model running predictions on a
minibatch, before training a single epoch on that batch. The model itself is a hybrid
approach for anomaly detection, using a simple autoencoder paired with a dynamic
threshold to map the anomaly score to a binary label. Finally, the prediction results
are evaluated using a sliding window confusion matrix along its anomaly detection
evaluation metrics (e.g. Precision, Recall, F1-score, etc.).

Note that this demonstration client can also be launched as a standalone detection
component, if no additional client is run along with the model aggregation server.
The same is the case for additional prediction and evaluation result aggregation
using centralize servers (see -h for more information).
However, the full demonstration topology consists of two federated IDS detection
clients along three servers (from the 'generic_fids_components' scripts subpackage):

    * model_aggr_server - Model aggregation server to aggregate the clients' models.
    * pred_aggr_server - Value aggregation server for (client) prediction results.
    * eval_aggr_server - Value aggregation server for (client) evaluation results.

ALl of these components, like with this client, can be found in the
'generic_fids_components' subpackage, to be launched directly through python,
beside the command line option. Note that one does not need to launch all three,
depending on the type of demo, one can select one, two, or all three additional
components.

There are also different options to choose pFL at startup.

Author: Fabian Hofmann, Seraphin Zunzer
Modified: 02.03.25
"""

import argparse
import logging
import pathlib

import numpy as np
import tensorflow as tf

from daisy.data_sources import (
    DataHandler,
    PcapDataSource,
    PysharkProcessor,
    pcap_f_features,
    pcap_nn_aggregator,
    demo_202303_label_data_point,
)
from daisy.evaluation import ConfMatrSlidingWindowEvaluation
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
        choices=[2, 5],
        required=True,
        help="ID of client (decides which data to draw from set)",
    )
    parser.add_argument(
        "--pcapBasePath",
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
        "--predServ",
        default="0.0.0.0",
        metavar="",
        help="IP or hostname of prediction server",
    )
    server_options.add_argument(
        "--predServPort",
        type=int,
        default=8002,
        choices=range(1, 65535),
        metavar="",
        help="Port of prediction server",
    )

    client_options = parser.add_argument_group("Client Options")
    client_options.add_argument(
        "--batchSize",
        type=int,
        default=256,
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
    pred_serv = None
    if args.predServ != "0.0.0.0":
        pred_serv = (args.predServ, args.predServPort)

    # Datasource
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
            pred_serv,
            input_size=65,
            label_split=65,
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
            pred_serv,
            input_size=65,
            label_split=65,
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
            pred_serv,
            input_size=65,
            label_split=65,
        )

    if args.pflMode == "layerwise":
        raise NotImplementedError


if __name__ == "__main__":
    create_client()
