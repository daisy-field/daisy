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
from src.federated_ids_components import FederatedOnlineClient
from src.federated_learning import TFFederatedModel, FederatedIFTM, EMAvgTM


def start_demo_client(client_id: int, pcap_dir_base_path: pathlib.Path, batch_size: int,
                      m_aggr_server: tuple[str, int], eval_server: tuple[str, int], aggr_server: tuple[str, int]):
    """TODO

    """
    handler = PcapHandler(f"{pcap_dir_base_path}diginet-cohda-box-dsrc{client_id}")
    processor = CohdaProcessor(client_id=client_id, events=march23_events)
    data_source = DataSource(source_handler=handler, data_processor=processor)

    id_fn = TFFederatedModel.get_fae(input_size=65, batch_size=batch_size)
    t_m = EMAvgTM(alpha=0.05)
    err_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    model = FederatedIFTM(identify_fn=id_fn, threshold_m=t_m, error_fn=err_fn, param_split=65)

    metrics = None
    # TODO EVAL METRICS

    client = FederatedOnlineClient(data_source=data_source, batch_size=batch_size, model=model,
                                   m_aggr_server=m_aggr_server, eval_server=eval_server, aggr_server=aggr_server,
                                   update_interval_t=batch_size * 8)
    client.start()


if __name__ == "__main__":
    # :param eval_server: Address of centralized evaluation server (see evaluator.py).
    #         :param aggr_server: Address of centralized aggregation server (see aggregator.py).
    #
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--debug", type=bool, default=False, help="Show debug outputs")
    parser.add_argument("--clientId", type=int, choices=[2, 5],
                        help="")  # TODO)
    parser.add_argument("--pcapBasePath", type=pathlib.Path,
                        default="../../../../Datasets/v2x_2023-03-06/",
                        help="Path to ")  # TODO
    parser.add_argument("--modelAggrServIp", default="0.0.0.0",
                        help="IP of evaluation server")
    parser.add_argument("--evalServPort", type=int, default=8001,
                        help="Port of evaluation server")
    parser.add_argument("--evalServIp", default="0.0.0.0",
                        help="IP of evaluation server")
    parser.add_argument("--evalServPort", type=int, default=8002,
                        help="Port of evaluation server")
    parser.add_argument("--evalServIp", default="0.0.0.0",
                        help="IP of evaluation server")
    parser.add_argument("--evalServPort", type=int, default=8003,
                        help="Port of aggregation server")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                            level=logging.DEBUG)
    else:
        logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                            level=logging.INFO)
