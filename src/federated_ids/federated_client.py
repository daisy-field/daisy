#
#   Created: 2023
#   Author: Seraphin Zunzer
#   License: MIT License
# ---------------------------------------------------------------



import socket
import threading
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
import keras
import sys, time
from datetime import datetime, timedelta
import logging
from src.federated_ids import federated_model as fm
import src.communication.message_stream as ms
from src.data_sources import *


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

    def __init__(self,  addr: Tuple[str, int], remote_addr: Tuple[str, int], eval_addr: Tuple[str, int],
                 data_source: DataSource, input_vector_size: int = 70, threshold_prediction: int = 2.2,
                 data_batchsize: int=256):

        self.train_data_batch = []  # container for data that should be trained with
        self.train_data_label = []


        self.data_container = []
        self.label_container = []
        self.time_container = []

        self._addr = addr
        self._remote_addr = remote_addr
        self._data_batchsize = data_batchsize
        self._input_vector_size = input_vector_size
        self._threshold_prediction = threshold_prediction
        self._eval_addr = eval_addr
        self._data_sorce = data_source
        self._global_weights = []
        self._train_count = 0

        # initialize prediction model and thread
        input_format = keras.layers.Input(shape=(input_size,))
        enc, dec = fm.FederatedModel.build()
        self.autoencoder = tf.keras.models.Model(inputs=input_format, outputs=dec(enc(input_format)))
        self.autoencoder.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='mse', metrics=[])

        #fm.FederatedModel.create_model()
        threading.Thread.__init__(self)

    def training(self):
        while 1:
            try:
                self._global_weights = _agg_endpoint.recv(300)
            except socket.timeout:
                logging.warning("Server not available")
                continue

            if len(train_data_batch) >= 1:
                train_dataset = np.array([item for sublist in data_thread.train_data_batch for item in sublist])
                logging.info(f"Start training on client {CLIENT_ID} with {len(train_dataset)} samples")
                self.local_weights = self.client_update(self._global_weights, train_dataset)
                data_thread.train_data_label = []
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

    def client_update(self, old_server_weights: [], dataset:[]):
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

    def run(self):
        """Collect samples until the data batch size is reached. Save this batch

        :return: -

        """
        logging.info(f"START CLIENT {self._addr}")

        # create data receiving thread
        data_thread = Data_Receiving_Thread()
        data_thread.start()

        _agg_endpoint = ms.EndpointSocket(addr=self._addr, remote_addr=self._remote_addr)
        _eval_endpoint = ms.EndpointSocket(addr=self._addr, remote_addr=self._eval_addr)

        t = Thread(target=training,  name="Training", daemon=True)
        t.start()

        self._data_sorce.open()
        for i in self._data_sorce:
            self.data_container.append(i)
            self.label_container.append(i)
            self.time_container.append(datetime.now())
            if len(self.data_container) > data_batchsize:
                # start prediction
                self.start_prediction()

                # empty containers
                self.data_container = []
                self.label_container = []
                self.time_container = []


    def start_prediction(self):
        """If Model is trained and data for prediction is available start prediction, otherwise train model

        :return: -
        """
        # if enough trainingrounds made
        if train_count > 0:

            # get datasets and make prediction
            pred_dataset = np.array(self.data_container)
            pred_labels = np.array(self.label_container)
            pred_time = np.array(self.time_container)
            self.make_predictions(pred_dataset, pred_labels, pred_time)

            # append data and labels to training data
            self.train_data_batch.append(pred_dataset)
            self.train_data_label.append(pred_labels)
        else:
            self.train_data_batch.append(self.data_container)
            self.train_data_label.append(self.label_container)

    def mad_score(self, points):
        """

        :param points:
        :return:
        """
        m = np.median(points)
        ad = np.abs(points - m)
        mad = np.median(ad)
        return 0.6745 * ad / mad

    def make_predictions(self, dataset, true_labels, pred_time):
        """Make predictions on the last databatch with latest global model weights.

        :param prediction_weights:
        :param dataset:
        :param true_labels:
        :param pred_time:
        :return:
        """

        self.autoencoder.set_weights(GLOBAL_WEIGHTS)

        try:
            reconstructions = self.autoencoder.predict(dataset)
            mse = np.mean(np.power(dataset - reconstructions, 2), axis=1)

        except Exception as e:
            logging.error(e)
            logging.error("Failed prediction")
            return

        z_scores = self.mad_score(mse)


        outliers = z_scores > threshold_prediction

        # get lists of the predicted and the true labels
        prediction = [self.anomaly if l else self.normal" for l in outliers]
        labels_true = ["BENIGN" if k == "BENIGN" else self.anomaly for k in true_labels]

        # calculate the time needed for all traffic
        time_needed = [(datetime.now() - pred_time[j]).total_seconds() for j in range(len(true_labels))]

        fp, tp, fn, tn = self.confusion_matrice(prediction, labels_true)

        self.metrics = [x + y for x, y in zip(self.metrics, [fp, tp, fn, tn])]

        logging.info(
            f'Client {CLIENT_ID}: False positives: {fp}, False negatives: {fn},'
            f' True positives: {tp}, True negatives: {tn}')

        #self.store_anomalies(labels_true, z_scores, time_needed, outliers)
        #self.analyze_MAD(z_scores, labels_true)


    def store_anomalies(self, true_labels:[], z_scores:[], time_needed:[], outliers:[]):
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


    class metrics_object():
        """

        """

        def __init__(self, prediction:[], true_labels:[], anomaly: String, normal: String):
            """Calculate confusion matrice

            :return: False positives, True positives, False negatives, True negatives
            """

            print("Evaluation Object created")
            self.prediction = prediction
            self.true_labels = true_labels
            self.normal = normal
            self.anomaly = anomaly
            fp = 0
            tp = 0
            fn = 0
            tn = 0
            for i in range(0, len(self.prediction)):
                if self.prediction[i] == self.anomaly and self.true_labels[i] == self.normal:
                    fp += 1
                elif self.prediction[i] == self.anomaly and self.true_labels[i] == self.anomaly:
                    tp += 1
                elif self.prediction[i] == self.normal and self.true_labels[i] == self.anomaly:
                    fn += 1
                elif self.prediction[i] == self.normal and self.true_labels[i] == self.normal:
                    tn += 1
            self.metrics=[fp, tp, fn, tn]

        def get_confusion_matrix(self):
            """Getter for confusion matrix

            :return: List: [False Positives, True Positives, False Negatives, True Negatives]
            """
            return self.metrics

        def false_positive_r(self):
            """ Calculate False Positive Rate

            :return: False Positive Rate
            """
            fp = self.metrics[0]
            tn = self.metrics[3]
            return (fp + tn) and fp / (fp + tn)

        def true_positive_r(self):
            """ Calculate True Positive Rate

            :return: True Positive Rate
            """
            tp = self.metrics[1]
            fn = self.metrics[2]
            return (tp + fn) and tp / (tp + fn)


        def accuracy(self):
            """ Calculate Accuracy

            :return: Accuracy
            """
            fp = self.metrics[0]
            tp = self.metrics[1]
            fn = self.metrics[2]
            tn = self.metrics[3]
            return (tp + tn) / (tp + tn + fp + fn)


    def process_anomalys(self, predictions:[], true_labels:[]):
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





