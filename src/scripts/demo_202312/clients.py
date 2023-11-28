"""TODO

    Author: Fabian Hofmann
    Modified: 28.11.23
"""

import argparse
import logging
import pathlib

import tensorflow as tf

from src.data_sources import DataSource
from src.data_sources import PcapHandler, CohdaProcessor, march23_events
from src.evaluation import ConfMatrSlidingWindowEvaluation
from src.federated_ids_components import FederatedOnlineClient
from src.federated_learning import TFFederatedModel, FederatedIFTM, EMAvgTM


def start_demo_client(client_id: int, pcap_dir_base_path: pathlib.Path, batch_size: int,
                      m_aggr_server: tuple[str, int], eval_server: tuple[str, int], aggr_server: tuple[str, int]):
    """TODO COMMENTS CHECKING

    """
    handler = PcapHandler(f"{pcap_dir_base_path}diginet-cohda-box-dsrc{client_id}")
    processor = CohdaProcessor(client_id=client_id, events=march23_events)
    data_source = DataSource(source_handler=handler, data_processor=processor)

    id_fn = TFFederatedModel.get_fae(input_size=65, batch_size=batch_size)
    t_m = EMAvgTM(alpha=0.05)
    err_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    model = FederatedIFTM(identify_fn=id_fn, threshold_m=t_m, error_fn=err_fn, param_split=65)

    metrics = [ConfMatrSlidingWindowEvaluation(window_size=batch_size * 32)]

    client = FederatedOnlineClient(data_source=data_source, batch_size=batch_size, model=model, metrics=metrics,
                                   m_aggr_server=m_aggr_server, eval_server=eval_server, aggr_server=aggr_server,
                                   update_interval_t=batch_size * 8)
    client.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--debug", type=bool, default=False, help="Show debug outputs")
    parser.add_argument("--clientId", type=int, choices=[2, 5],
                        help="")  # TODO)
    parser.add_argument("--pcapBasePath", type=pathlib.Path,
                        default="../../../../Datasets/v2x_2023-03-06/",
                        help="Path to ")  # TODO
    parser.add_argument("--batchSize", default=32)
    parser.add_argument("--modelAggrServIp", default="0.0.0.0",
                        help="IP of model aggregation server")
    parser.add_argument("--modelAggrServPort", type=int, default=8001,
                        help="Port of model aggregation server")
    parser.add_argument("--noEvalServ", type=bool, default=False, help="Disables centralized evaluation")
    parser.add_argument("--evalServIp", default="0.0.0.0",
                        help="IP of evaluation server")
    parser.add_argument("--evalServPort", type=int, default=8002,
                        help="Port of evaluation server")
    parser.add_argument("--noAggrServ", type=bool, default=False, help="Disables centralized aggregation")
    parser.add_argument("--aggrServIp", default="0.0.0.0",
                        help="IP of aggregation server")
    parser.add_argument("--aggrServPort", type=int, default=8003,
                        help="Port of aggregation server")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                            level=logging.DEBUG)
    else:
        logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                            level=logging.INFO)

    m_aggr_serv = (args.modelAggrServIp, args.modelAggrServPort)
    eval_serv = None
    if not args.noEvalServ:
        eval_serv = (args.evalServIp, args.evalServPort)
    aggr_serv = None
    if not args.noEvalServ:
        aggr_serv = (args.aggrServIp, args.aggrServPort)

    start_demo_client(client_id=args.clientId, pcap_dir_base_path=args.pcapBasePath, batch_size=args.batchSize,
                      m_aggr_server=m_aggr_serv, eval_server=eval_serv, aggr_server=aggr_serv)
