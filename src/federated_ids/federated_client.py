#
#   Created: 2023
#   Author: Seraphin Zunzer
#   License: MIT License
# ---------------------------------------------------------------


import logging
import socket
import threading
from datetime import datetime
from typing import Tuple

import numpy as np
from tensorflow.keras import backend as K

import src.communication.message_stream as ms
import src.data_sources.data_source as ds
import src.federated_ids.metrics as metrics
from src.federated_ids import federated_model as fm

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

        self.train_data_batch = []  # container for data that should be trained with
        self.train_data_label = []

        self.data_queue = []
        self.label_queue = []
        self.time_queue = []

        self._addr = addr
        self._agg_addr = agg_addr
        self._eval_addr = eval_addr

        self._data_batchsize = train_batchsize
        self._threshold_prediction = threshold_prediction
        self._data_sorce = data_source
        self._global_weights = []
        self._train_count = 0

        self.data_batch_queue = []
        self.label_batch_queue = []

        self._prediction_model = federated_model.compile_model()

        threading.Thread.__init__(self)

    def run(self):
        """Collect samples until the data batch size is reached. Save this batch

        :return: -

        """
        logging.info(f"START CLIENT {self._addr}")

        # create data receiving thread
        data_thread = Data_Receiving_Thread()
        data_thread.open()

        _agg_endpoint = ms.EndpointSocket(addr=self._addr, remote_addr=self._agg_addr)
        _eval_endpoint = ms.EndpointSocket(addr=self._addr, remote_addr=self._eval_addr)

        t = Thread(target=training, name="Training", daemon=True)
        t.open()

        self._data_sorce.open()
        for i in self._data_sorce:
            self.data_queue.append(i)
            self.label_queue.append("Normal")
            self.time_queue.append(datetime.now())
            if len(self.data_queue) > self._data_batchsize:
                self.start_prediction(_eval_endpoint)

                # empty containers
                self.data_queue = []
                self.label_queue = []
                self.time_queue = []

    def start_prediction(self, _eval_endpoint: ms.StreamEndpoint):
        """If Model is trained and data for prediction is available start prediction, otherwise train model

        :return: -
        """

        if train_count > 0:
            self.make_predictions(_eval_endpoint, np.array(self.data_queue), np.array(self.label_queue),
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
        self._prediction_model.set_weights(GLOBAL_WEIGHTS)
        try:
            reconstructions = self._prediction_model.predict(dataset)
            mse = np.mean(np.power(dataset - reconstructions, 2), axis=1)
        except Exception as e:
            logging.error(e)
            logging.error("Failed prediction")
            return

        z_scores = self.mad_score(mse)

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

    def mad_score(self, points):
        """
        :param points:
        :return:
        """
        m = np.median(points)
        ad = np.abs(points - m)
        mad = np.median(ad)
        return 0.6745 * ad / mad

    def store_anomalies(self, true_labels: [], z_scores: [], time_needed: [], outliers: []):
        """Store times needed for processing, the result of classification and the calculated score in specific txt file

        :param true_labels:
        :param z_scores:
        :param time_needed:
        :param outliers:
        :return:
        """
        with open(f'results/times/times_[Installation Attack Tool].txt', "a+", newline='') as ia:
            with open(f'results/times/times_[SSH Brute Force].txt', "a+", newline='') as bf:
                with open(f'results/times/times_[SSH Privilege Escalation].txt', "a+", newline='') as pe:
                    with open(f'results/times/times_[SSH Brute Force Response].txt', "a+", newline='') as br:
                        with open(f'results/times/times_[SSH  Data leakage].txt', "a+", newline='') as dl:
                            for i in range(0, len(true_labels)):
                                if true_labels[i][0] == "Installation Attack Tool":
                                    ia.write(f'{time_needed[i]}, {outliers[i]}, {round(z_scores[i], 7)}\n')
                                if true_labels[i][0] == "SSH Brute Force":
                                    bf.write(f'{time_needed[i]}, {outliers[i]}, {round(z_scores[i], 7)}\n')
                                if true_labels[i][0] == "SSH Privilege Escalation":
                                    pe.write(f'{time_needed[i]}, {outliers[i]}, {round(z_scores[i], 7)}\n')
                                if true_labels[i][0] == "SSH Brute Force Response":
                                    br.write(f'{time_needed[i]}, {outliers[i]}, {round(z_scores[i], 7)}\n')
                                if true_labels[i][0] == "SSH  Data leakage":
                                    dl.write(f'{time_needed[i]}, {outliers[i]}, {round(z_scores[i], 7)}\n')

    def analyze_MAD(self, z_scores, labels_true):
        """Analyze MAD threshold. Set diiferent thresholds and calculate True positive rate

        :param z_scores:
        :param labels_true:
        :return:
        """
        tpr_i = []
        i = 0
        while i < 10:
            outliers = z_scores > i
            pred = ["BENIGN" if not l else "ANOMALY" for l in outliers]
            fp, tp, fn, tn = self.confusion_matrice(pred, labels_true)
            fpr = fp / (fp + tn)
            tpr_i.append(fpr)
            i += 0.1
        with open('results/mad.txt', 'w') as file:
            for i, tpr in enumerate(tpr_i):
                file.write(f'{round(i * 0.1, 2)}	{tpr}\n')

    def training(self):
        while 1:
            try:
                self._global_weights = _agg_endpoint.recv(300)
            except socket.timeout:
                logging.warning("Server not available")
                continue

            if len(self.data_batch_queue) >= 1:
                train_dataset = np.array([item for sublist in self.data_batch_queue for item in sublist])
                logging.info(f"Start training on client {CLIENT_ID} with {len(train_dataset)} samples")
                self.local_weights = self.client_update(self._global_weights, train_dataset)
                self.data_batch_queue = []
                self.label_batch_queue = []

                data_thread.train_data_batch = []

                try:
                    _agg_endpoint.send(self._local_weights)
                except:
                    logging.error("Could not send weights back")

                logging.info("Finished training, closed session")
                K.clear_session()  # clear model
            else:
                logging.warning("No data available for training")
                _agg_endpoint.send("No data")

    def client_update(self, old_server_weights: [], dataset: []):
        """
        Performs training using the received server model weights on the client's dataset

        :param dataset: current dataset for training in this round
        :return: new weights after training
        """
        model = fm.FederatedModel.create_model()
        model.set_weights(old_server_weights)
        history = model.fit(dataset, dataset, verbose=1, epochs=0, batch_size=32)
        weights = model.get_weights()  # get new weights
        return weights

    def process_anomalys(self, predictions: [], true_labels: []):
        """Function to process anomalies, e.g. delete packets, throw alerts etc.
        In this case write anomaly to file with timestamp.

        :param predictions:
        :param true_labels:
        :return:
        """
        now = datetime.now()
        timestamp = now.strftime("%m/%d/%Y, %H:%M:%S")

        with open(f"results_{CLIENT_ID}.txt", "a") as txt_file:
            for i in range(len(predictions)):
                if predictions[i] == "anomaly":
                    txt_file.write(f" {timestamp} - {true_labels[i]}  \n")


if __name__ == "__main__":
    client = Client(("127.0.0.1", 54321), ("127.0.0.1", 54322), ("127.0.0.1", 54323))  # TODO fm.FederatedModel())
    client.run()
