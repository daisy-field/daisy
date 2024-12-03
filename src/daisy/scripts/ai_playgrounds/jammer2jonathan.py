import argparse
import logging

from daisy.communication import StreamEndpoint
from daisy.data_sources import (
    CSVFileDataSource,
    DataProcessor,
    DataHandler,
    DataHandlerRelay,
    pcap_nn_aggregator
)
import tensorflow as tf

from daisy.evaluation import ConfMatrSlidingWindowEvaluation
from daisy.federated_ids_components import FederatedOnlineClient
from daisy.federated_learning import TFFederatedModel, FederatedIFTM, EMAvgTM


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
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

    parser.add_argument(
        "--input-file",
        "-f",
        type=str,
        metavar="FILE",
        required=True,
        help="The input CSV file to read.",
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


def create_relay():
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
    data_source = CSVFileDataSource(
        files=args.input_file, name="JammerClient:DataSource"
    )
    data_processor = DataProcessor().dict_to_array(nn_aggregator=pcap_nn_aggregator)
    data_handler = DataHandler(
        data_source=data_source,
        data_processor=data_processor,
        name="JammerClient:DataHandler",
        multithreading=args.handler_multithreading,
        buffer_size=args.handler_buffer_size,
    )

    # Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.MeanAbsoluteError()
    id_fn = TFFederatedModel.get_fae(
        input_size=65,
        optimizer=optimizer,
        loss=loss,
        batch_size=args.batchSize,
        epochs=1,
    )
    t_m = EMAvgTM(alpha=0.05)
    err_fn = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    model = FederatedIFTM(identify_fn=id_fn, threshold_m=t_m, error_fn=err_fn)

    metrics = [ConfMatrSlidingWindowEvaluation(window_size=args.batchSize * 8)]

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
    )
    client.start()
    input("Press Enter to stop client...")
    client.stop()


if __name__ == "__main__":
    create_relay()
