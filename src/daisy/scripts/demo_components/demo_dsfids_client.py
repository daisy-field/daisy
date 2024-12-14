# Copyright (C) 2024 DAI-Labor and others
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

Author: Fabian Hofmann
Modified: 04.11.24
"""

import argparse
import logging
import pathlib

import numpy as np
import tensorflow as tf

from daisy.data_sources import (
    DataHandler,
    PysharkProcessor,
    pcap_nn_aggregator,
)
from daisy.evaluation import ConfMatrSlidingWindowEvaluation
from daisy.federated_ids_components import FederatedOnlineClient
from daisy.federated_learning import TFFederatedModel, FederatedIFTM

from daisy.personalized_fl_components.generative.generative_model import GenerativeGAN
from daisy.personalized_fl_components.auto_model_scaler import AutoModelScaler
from daisy.personalized_fl_components.generative.generative_node import (
    pflGenerativeNode,
)
from daisy.personalized_fl_components.distillative.distillative_node import (
    pflDistillativeNode,
)

from daisy.federated_learning import SMAvgTM

from daisy.data_sources import CSVFileDataSource
from daisy.data_sources.network_traffic.dsfids import (
    dsfids_label_data_point,
    dsfids_f_features,
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
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
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

    # Datasource
    source = CSVFileDataSource(f"{args.pcapBasePath}/Host-{args.clientId}")
    processor = (
        PysharkProcessor()
        # .packet_to_dict()
        .select_dict_features(features=dsfids_f_features, default_value=np.nan)
        .add_func(lambda o_point: dsfids_label_data_point(d_point=o_point))
        .dict_to_array(nn_aggregator=pcap_nn_aggregator)
    )

    data_handler = DataHandler(data_source=source, data_processor=processor)
    t_m = SMAvgTM()
    err_fn = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

    metrics = [ConfMatrSlidingWindowEvaluation(window_size=args.batchSize * 8)]

    if args.pflMode is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = tf.keras.losses.MeanAbsoluteError()
        id_fn = TFFederatedModel.get_fae(
            input_size=65,
            optimizer=optimizer,
            loss=loss,
            batch_size=args.batchSize,
            epochs=1,
        )
        model = FederatedIFTM(identify_fn=id_fn, threshold_m=t_m, error_fn=err_fn)

        # Client
        client = FederatedOnlineClient(
            data_handler=data_handler,
            batch_size=args.batchSize,
            model=model,
            label_split=65,
            metrics=metrics,
            m_aggr_server=m_aggr_serv,
            eval_server=eval_serv,
            aggr_server=aggr_serv,
            update_interval_t=args.updateInterval,
            poisoning_mode=args.poisoningMode,
        )
        client.start()
        input("Press Enter to stop client...")
        client.stop()

    if args.pflMode == "generative":
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = tf.keras.losses.MeanAbsoluteError()
        id_fn = None
        input_size = 65
        epochs = 1
        ams = AutoModelScaler()

        if args.autoModel:
            id_fn = ams.choose_model(
                input_size, optimizer, loss, args.batchSize, epochs
            )
        if not args.autoModel:
            id_fn = ams.get_manual_model(
                args.manualModel, input_size, optimizer, loss, args.batchSize, epochs
            )

        model = FederatedIFTM(identify_fn=id_fn, threshold_m=t_m, error_fn=err_fn)

        generative_gan = GenerativeGAN.create_gan(
            input_size=65,
            discriminator_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            generator_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        )

        # Client
        client = pflGenerativeNode(
            data_handler=data_handler,
            batch_size=args.batchSize,
            model=model,
            label_split=65,
            metrics=metrics,
            m_aggr_server=m_aggr_serv,
            eval_server=eval_serv,
            aggr_server=aggr_serv,
            update_interval_t=args.updateInterval,
            generative_model=generative_gan,
            poisoning_mode=args.poisoningMode,
        )
        client.start()
        input("Press Enter to stop client...")
        client.stop()

    if args.pflMode == "distillative":
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = tf.keras.losses.MeanAbsoluteError()
        id_fn = None
        input_size = 65
        epochs = 1
        aMS = AutoModelScaler()

        if args.autoModel:
            id_fn = aMS.choose_model(
                input_size, optimizer, loss, args.batchSize, epochs
            )
        if not args.autoModel:
            id_fn = aMS.get_manual_model(
                args.manualModel, input_size, optimizer, loss, args.batchSize, epochs
            )

        model = FederatedIFTM(identify_fn=id_fn, threshold_m=t_m, error_fn=err_fn)

        # Client
        client = pflDistillativeNode(
            data_handler=data_handler,
            batch_size=args.batchSize,
            model=model,
            label_split=65,
            metrics=metrics,
            m_aggr_server=m_aggr_serv,
            eval_server=eval_serv,
            aggr_server=aggr_serv,
            update_interval_t=args.updateInterval,
            poisoning_mode=args.poisoningMode,
        )
        client.start()
        input("Press Enter to stop client...")
        client.stop()

    if args.pflMode == "layerwise":
        raise NotImplementedError

    input("Press Enter to stop client...")
    client.stop()


if __name__ == "__main__":
    create_client()
