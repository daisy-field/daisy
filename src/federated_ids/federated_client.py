"""
    Class for Federated Client

    Author: Seraphin Zunzer
    Modified: 09.08.23
"""


import logging
import socket
import threading
from datetime import datetime
from typing import Tuple

import numpy as np
from tensorflow.keras import backend as K

import src.communication.message_stream as ms
import src.data_sources.data_source as ds
import utils.metrics as metrics
from federated_model import federated_model as fm
from utils.mad_score import calculate_mad_score

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.DEBUG)


class Client:
    """
        Class to create a single federated client. Parameters ID, Port and Receiving Port are received from arguments
        Optimized client for real-world dataset.
        Federated learning client that receives the global model weights, detects attacks in the arriving real-world
        datastreams from the data simulator, and trains the local model afterward.
        This model is transmitted to the Server for weight aggregation using FedAVG

        :args: ClientID ClientPort DataReceivingPort
        """

    def __init__(self, addr: Tuple[str, int], agg_addr: Tuple[str, int], eval_addr: Tuple[str, int],
                 data_source: ds.DataSource, federated_model: fm.FederatedModel, threshold_prediction: int = 2.2,
                 train_batchsize: int = 256):
        """

        :param addr: Address of this client
        :param agg_addr: Address of aggregation server
        :param eval_addr: Address of evaluation server
        :param data_source: Datasource
        :param federated_model: Federated Model
        :param threshold_prediction: Prediction threshold
        :param train_batchsize: Batchsize of training
        """
        self.train_data_batch = []  # container for data that should be trained with
        self.train_data_label = []

        self.data_queue = [] # container for incoming data from datasource
        self.label_queue = []
        self.time_queue = []


        self._data_batchsize = train_batchsize
        self._threshold_prediction = threshold_prediction
        self._data_sorce = data_source
        self._train_count = 0
        self._addr= addr
        self.data_batch_queue = []
        self.label_batch_queue = []

        self._model = federated_model
        self._prediction_model = federated_model
        self._agg_endpoint = ms.StreamEndpoint(name="Aggregator", addr=addr, remote_addr=agg_addr)
        self._eval_endpoint = ms.StreamEndpoint(name="Evaluator", addr=addr, remote_addr=eval_addr)

    def start_training(self):
        while 1:
            try:
                global_weights = self._agg_endpoint.receive(300)
            except socket.timeout:
                logging.warning("Server not available")
                continue

            if len(self.data_batch_queue) >= 1:
                train_dataset = np.array([item for sublist in self.data_batch_queue for item in sublist])
                logging.info(f"Start training on client {self._addr} with {len(train_dataset)} samples")
                self._model.build_model()
                self._model.compile_model()
                self._model.set_model_weights(global_weights)
                self._model.fit_model(x=train_dataset, y=train_dataset, verbose=1, epochs=0, batch_size=32)

                self.data_batch_queue = []
                self.label_batch_queue = []

                try:
                    self._agg_endpoint.send(self._model.get_model_weights())
                except:
                    logging.error("Error in sending weights to aggregation server")

                logging.info("Finished training, closed session")
                K.clear_session()  # clear model
            else:
                logging.warning("No data available for training")
                self._agg_endpoint.send("No data")


    def run(self):
        """Collect samples until the data batch size is reached. Save this batch

        :return: -

        """
        logging.info(f"START CLIENT {self._addr}")


        t = threading.Thread(target=start_training, name="Training", daemon=True)
        t.open()

        #receive data from datasource
        self._data_sorce.open()
        for i in self._data_sorce:
            self.data_queue.append(i)
            self.label_queue.append("Normal")
            self.time_queue.append(datetime.now())
            if len(self.data_queue) > self._data_batchsize:
                self.start_prediction()

                # empty containers
                self.data_queue = []
                self.label_queue = []
                self.time_queue = []

    def start_prediction(self):
        """If Model is trained and data for prediction is available start prediction, otherwise train model

        :return: -
        """
        if self._train_count > 0:
            self.make_predictions(np.array(self.data_queue), np.array(self.label_queue),
                                  np.array(self.time_queue))

            self.data_batch_queue.append(pred_dataset)
            self.label_batch_queue.append(pred_labels)
        else:
            self.data_batch_queue.append(self.data_queue)
            self.label_batch_queue.append(self.label_queue)

    def make_predictions(self, endpoint: ms.StreamEndpoint, dataset: np.array, true_labels: np.array,
                         pred_time: np.array):
        """Make predictions on the last databatch with latest global model weights.

        :param prediction_weights:
        :param dataset:
        :param true_labels:
        :param pred_time:
        :return:
        """
        self._prediction_model.set_model_weights(self)
        try:
            reconstructions = self._prediction_model.predict(dataset)
            mse = np.mean(np.power(dataset - reconstructions, 2), axis=1)
        except Exception as e:
            logging.error(e)
            logging.error("Failed prediction")
            return

        z_scores = calculate_mad_score(mse)

        outliers = z_scores > threshold_prediction

        # get lists of the predicted and the true labels
        prediction = [self.anomaly if l else self.normal for l in outliers]
        labels_true = [self.normal if k == self.normal else self.anomaly for k in true_labels]

        time_needed = [(datetime.now() - pred_time[j]).total_seconds() for j in range(len(true_labels))]

        metrics_obj = metrics.MetricsObject(prediction, labels_true, "Benign", "Anomaly")

        logging.info(
            f'Client {self._addr}: False positives: {evaluation.metrics[0]}, True positives: {evaluation.metrics[1]},'
            f'False negatives: {evaluation.metrics[2]}, True negatives: {evaluation.metrics[3]}')

        # self.store_anomalies(labels_true, z_scores, time_needed, outliers)
        # self.analyze_MAD(z_scores, labels_true)

        endpoint.send(metrics_obj)



if __name__ == "__main__":
    client = Client(("127.0.0.1", 54321), ("127.0.0.1", 54322), ("127.0.0.1", 54323))  # TODO fm.FederatedModel())
    client.run()
