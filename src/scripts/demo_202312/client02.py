import logging

import tensorflow as tf

from src.data_sources import DataSource
from src.data_sources import PcapHandler, CohdaProcessor, march23_events

from src.federated_ids_components import FederatedOnlineClient
from src.federated_learning import TFFederatedModel, FederatedIFTM, EMAvgTM

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.DEBUG)

    handler = PcapHandler()#TODO
    processor = CohdaProcessor(client_id=2, events=march23_events, )#TODO
    data_source = DataSource(source_handler=handler, data_processor=processor)

    id_fn = TFFederatedModel.get_fae(input_size=10)
    t_m = EMAvgTM(alpha=0.05)
    err_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    model = FederatedIFTM(identify_fn=id_fn, threshold_m=t_m, error_fn=err_fn, param_split=10)
    # TODO
    client_01 = FederatedOnlineClient()
