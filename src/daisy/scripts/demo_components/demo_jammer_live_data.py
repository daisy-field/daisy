# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Pre-configured demonstration client for a federated Jamming Detection System (fJDS),
designed to collaboratively learn with other clients through a centralized model
aggregation server using the federated averaging (FedAvg) technique. This client
is specifically configured to process pre-processed 5G traffic data from
5G antenna infrastructure, focusing on the detection
of deliberate jamming attacks.

This demonstration client can operate either as part of a federated setup,
interacting with a model aggregation server and optionally prediction/evaluation
aggregation servers, or as a standalone detection system. The full demonstration
topology typically includes the following components from the `generic_fids_components`
subpackage:

    * model_aggr_server - Aggregates models from multiple clients.
    * pred_aggr_server - Aggregates prediction results from clients.
    * eval_aggr_server - Aggregates evaluation metrics from clients.

All these components, along with this client, can be found in the
`generic_fids_components` subpackage and can be launched via Python or the command line.
Depending on the specific demonstration setup, any combination of these components can
be utilized.

Author: Simon Torka
Modified: 04.11.24
"""

import argparse
import logging
from datetime import datetime

from daisy.communication import StreamEndpoint
from daisy.data_sources import (
    DataProcessor,
    DataHandler,
    pcap_nn_aggregator,
    JammerWebSocketDataSource,
    SimpleRemoteDataSource,
)
import tensorflow as tf

from daisy.evaluation import ConfMatrSlidingWindowEvaluation
from daisy.federated_ids_components import FederatedOnlineClient
from daisy.federated_learning import (
    TFFederatedModel,
    FederatedIFTM,
    EMAMADThresholdModel,
)

from keras.optimizers import Adam


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
    """Parses command-line arguments.

    Defines and organizes options for logging, server configurations, client settings,
    and performance configurations.

    :return: Parsed arguments as an argparse.Namespace object.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--url",
        type=str,
        default="ws://172.21.6.70:4501/Phy",  # "ws://10.42.6.111:4501/Phy",
        help="WebSocket-URL des Servers (Default: ws://10.42.6.111:4501/Phy)",
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
        default=127,  # 32,
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

    performance_group = parser.add_argument_group(
        "Performance Configuration", "These arguments can adjust performance."
    )
    performance_group.add_argument(
        "--handler-multithreading",
        "-hmt",
        action="store_true",
        help="Enables multithreading for the data handler.",
    )
    performance_group.add_argument(
        "--handler-buffer-size",
        "-hbs",
        type=int,
        metavar="BUFFER_SIZE",
        default=1024,
        help="The buffer size for the data handler.",
    )

    return parser.parse_args()


def check_args(args):
    """Configures logging based on the parsed arguments.

    :param args: Parsed command-line arguments.
    """
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
            level=20,  # log_level,
        )
    else:
        logging.basicConfig(
            filename=args.log_file,
            filemode="w",
            format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=log_level,
        )


def create_data_source(args):
    """Creates and returns the data handler"""

    if args.url:
        return JammerWebSocketDataSource(
            name="jamming_data_collector:jamming_live_data_source",
            url=args.url,
            log_level=args.loglevel,
        )
    else:
        stream_endpoint = StreamEndpoint(
            name="data_collector:stream_endpoint",
            addr=(args.local_ip, args.local_port),
            remote_addr=(args.remote_ip, args.remote_port),
            acceptor=True,
            multithreading=args.io_multithreading,
        )
        return SimpleRemoteDataSource(
            endpoint=stream_endpoint, name="data_collector:remote_data_source"
        )


def create_relay():
    """Creates and starts the federated client relay.

    This function initializes the data sources, models, metrics, and client configuration.
    """
    # Args parsing
    args = _parse_args()
    check_args(args)

    m_aggr_serv = (args.modelAggrServ, args.modelAggrServPort)
    eval_serv = None
    if args.evalServ != "0.0.0.0":
        eval_serv = (args.evalServ, args.evalServPort)
    aggr_serv = None
    if args.aggrServ != "0.0.0.0":
        aggr_serv = (args.aggrServ, args.aggrServPort)

    # Datasource
    data_source = create_data_source(args)

    features_to_keep = [
        # "Jammer_On",
        "CQI_DLLA",
        "MCS_DLLA",
        "CQI_Change_DLLA",
        "Nack_DLLA",
        "Alloc_RBs_DLLA",
        # "UL-BLER-CRC_UELink",
        "C2I_UELink",
        #   "Connected_UELink",
        "Layers_DLLA",
        "RI_Change_DLLA",
        "Margin_ChangeM_DLLA",
        "DL_Aggregation_Level_1_DLLA",
        "DLAL_2_DLLA",
        "DLAL_4_DLLA",
        "DLAL_8_DLLA",
        "DLAL_16_DLLA",
        # "UL-MCS_UELink",
        "UL_BLER_CRC_UELink",
    ]

    data_processor = (
        DataProcessor()
        .keep_dict_feature(features_to_keep)
        .dict_to_array(nn_aggregator=pcap_nn_aggregator)
    )
    data_handler = DataHandler(
        data_source=data_source,
        data_processor=data_processor,
        name="JammerClient:DataHandler",
        multithreading=args.handler_multithreading,
        buffer_size=args.handler_buffer_size,
    )

    # test
    # data_source.open()

    # for sample in data_source:
    #    print(sample)

    # Model
    optimizer = Adam(learning_rate=0.001)

    id_fn = TFFederatedModel.get_fvae(
        input_size=15,
        latent_dim=8,
        hidden_layers=[32, 16],
        optimizer=optimizer,
        batch_size=args.batchSize,
        epochs=1,
        metrics=["accuracy"],
    )
    t_m = EMAMADThresholdModel(alpha=0.2, mad_multiplier=2.5)  # EMAvgTM(alpha=0.05)
    err_fn = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    model = FederatedIFTM(identify_fn=id_fn, threshold_m=t_m, error_fn=err_fn)

    metrics = [ConfMatrSlidingWindowEvaluation(window_size=args.batchSize * 8)]

    # Client
    client = FederatedOnlineClient(
        data_handler=data_handler,
        batch_size=args.batchSize,  # z.B: 10*32
        model=model,
        label_split=15,
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
    create_relay()
