"""
    Class for Federated Client

    Author: Seraphin Zunzer
    Modified: 09.08.23
"""

import logging
import socket
import threading
from datetime import datetime
from time import sleep
from typing import Tuple

import numpy as np
from tensorflow.keras import backend as K

import src.communication.message_stream as ms
import src.data_sources.data_source as ds
import evaluation.anomaly_processing
import evaluation.metrics as metrics
from data_sources import PcapHandler, PysharkProcessor
from federated_learning import federated_model as fm
from federated_learning.TOBEREMOVEDmodels.autoencoder import FedAutoencoder
from evaluation.mad_score import calculate_mad_score

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
                 train_batchsize: int = 32, normal_label: str = "Benign", anomaly_label: str = "Anomaly",
                 epochs: int = 20):
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

        self.data_queue = []  # container for incoming data from datasource
        self.label_queue = []
        self.time_queue = []

        self._data_batchsize = train_batchsize
        self._threshold_prediction = threshold_prediction
        self._data_sorce = data_source
        self._train_count = 0
        self._addr = addr
        self._global_weights = []
        self.data_batch_queue = []
        self.label_batch_queue = []

        self._anomaly_label = anomaly_label
        self._normal_label = normal_label
        self._model = federated_model
        self._prediction_model = federated_model
        self._epochs = epochs
        logging.info(f"Starting Endpoints.")

        self._agg_endpoint = ms.StreamEndpoint(name="Agg_connection", acceptor=False, multithreading=False, addr=addr,
                                               remote_addr=agg_addr)
        self._agg_endpoint.start()
        # self._eval_endpoint = ms.StreamEndpoint(name="Eval_connection", acceptor=False, addr=addr, remote_addr=eval_addr)
        # self._eval_endpoint.start()
        logging.info(f"Client {addr} started.")

    def start_training(self):
        while 1:
            try:
                logging.info("Waiting for model weights")
                self._global_weights = self._agg_endpoint.receive()
                logging.info(f"Received new global model weights.")

            except socket.timeout:
                logging.warning("Server not available")
                continue

            if len(self.data_batch_queue) >= 1:
                train_dataset = np.array([item for sublist in self.data_batch_queue for item in sublist])
                print(train_dataset)
                logging.info(f"Prepare training on client {self._addr} with {len(train_dataset)} samples")
                self._model.init_model() # TODO WHY INIT EVERY ITERATION? ONLY FOR OPTIMIZERS, ETC NOT WEIGTHS
                self._model._model.set_weights(self._global_weights)
                logging.info("STARTED TRAINING")
                self._model._model.fit(x=train_dataset, y=train_dataset, epochs=self._epochs, verbose=1, batch_size=32)
                logging.info("TRAINING FINISHED")

                self.data_batch_queue = []
                self.label_batch_queue = []

                try:
                    self._agg_endpoint.send(self._model._model.get_weights())
                except:
                    logging.error("Error in sending weights to aggregation server")

                logging.info("Finished training, closed session")
                K.clear_session()  # clear model
            else:
                logging.warning("No data available for training")

    def run(self):
        """Collect samples until the data batch size is reached. Save this batch

        :return: -

        """

        # TODO DOES THIS NOT CREATE ONE BIG RACE CONDITION DURING PREDICTION MAKING?
        training_thread = threading.Thread(target=self.start_training, name="Training", daemon=True)
        training_thread.start()

        # receive data from datasource
        self._data_sorce.open()
        for i in self._data_sorce:
            self.data_queue.append(i)
            self.label_queue.append("Normal")
            self.time_queue.append(datetime.now())
            sleep(0.1)
            if len(self.data_queue) > self._data_batchsize:
                # self.start_prediction()
                self.data_batch_queue.append(self.data_queue)
                self.label_batch_queue.append(self.label_queue)
                logging.info(
                    f"One data batch filled. Currently {len(self.data_batch_queue)} batches for training available")

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

            self.data_batch_queue.append(self.data_queue)
            self.label_batch_queue.append(self.label_queue)
        else:
            self.data_batch_queue.append(self.data_queue)
            self.label_batch_queue.append(self.label_queue)

    def make_predictions(self, dataset: np.array, true_labels: np.array,
                         pred_time: np.array):
        """Make predictions on the last databatch with latest global model weights.

        :param dataset: dataset to make predictions on
        :param true_labels: true labels belonging to dataset
        :param pred_time: List of times, the data samples arrived
        :return:
        """
        self._prediction_model.set_model_weights(self._global_weights)
        try:
            reconstructions = self._prediction_model.model_predict(x=dataset)
            mse = np.mean(np.power(dataset - reconstructions, 2), axis=1)
        except Exception as e:
            logging.error(e)
            logging.error("Failed prediction")
            return

        z_scores = calculate_mad_score(mse)

        outliers = z_scores > self._threshold_prediction

        # get lists of the predicted and the true labels
        prediction = [self._anomaly_label if l else self._normal_label for l in outliers]
        labels_true = [self._normal_label if k == self._normal_label else self._anomaly_label for k in true_labels]

        time_needed = [(datetime.now() - pred_time[j]).total_seconds() for j in range(len(true_labels))]

        metrics_obj = metrics.MetricsObject(prediction, labels_true, self._anomaly_label, self._normal_label)

        logging.info(
            f'Client {self._addr}: False positives: {metrics_obj.metrics[0]}, True positives: {metrics_obj.metrics[1]},'
            f'False negatives: {metrics_obj.metrics[2]}, True negatives: {metrics_obj.metrics[3]}')

        evaluation.anomaly_processing.store_anomalies(labels_true, z_scores, time_needed, outliers)
        evaluation.mad_score.analyze_mad_score(z_scores, labels_true)

        self._eval_endpoint.send(metrics_obj)


# TODO MUST BE OUTSOURCED INTO A PROPER STARTSCRIPT FOR DEMO PURPOSES

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)

    d = ds.DataSource("test", source_handler=PcapHandler('test_data'), data_processor=PysharkProcessor())
    client = Client(("127.0.0.1", 54321), ("127.0.0.1", 54322), ("127.0.0.1", 54323), data_source=d,
                    federated_model=FedAutoencoder())
    client.run()
