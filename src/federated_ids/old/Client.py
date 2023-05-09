#
#   Created: 2023
#   Author: Seraphin Zunzer
#   License: MIT License
# ---------------------------------------------------------------

#


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
logging.basicConfig(level=logging.DEBUG)


HOST = "127.0.0.1"  # The server's hostname or IP address
SERVER_PORT = 65432  # The port of the server
CLIENT_PORT = None  # The port used by the client, set by cli
DATA_RECEIVING_PORT = None  # port for incoming data
CLIENT_ID = None  # will be set with command line arguments

encryption = True
data_batchsize = 256
input_size = 70
threshold_prediction = 2.2

global GLOBAL_WEIGHTS
global train_count


class FederatedModel:
    """Creates the keras model that is passed to the clients"""

    @staticmethod
    def build():
        """Build Keras Autoencoder

        :return: Encoder and Decoder
        """
        encoder = Sequential([
            keras.layers.Dense(input_size, input_shape=(input_size,)),
            keras.layers.Dense(35),
            keras.layers.Dense(18),
        ])
        decoder = Sequential([
            keras.layers.Dense(35, input_shape=(18,)),
            keras.layers.Dense(input_size),
            keras.layers.Activation("sigmoid"),
        ])
        return encoder, decoder


# -----------------------------------------  Data Receiving & Prediction Thread ----------------------------


class Data_Receiving_Thread(threading.Thread):
    """Class to create thread for receiving incoming data"""

    def __init__(self):
        self.cohda_host = "127.0.0.1"
        self.cohda_port = DATA_RECEIVING_PORT

        self.train_data_batch = []  # container for data that should be trained with
        self.train_data_label = []

        self.data_container = []
        self.label_container = []
        self.time_container = []

        # initialize socket
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket.bind((self.cohda_host, self.cohda_port))
        self.data_socket.listen()

        # initialize metrics
        self.metrics = [0, 0, 0, 0]

        # initialize prediction model and thread
        input_format = keras.layers.Input(shape=(input_size,))
        enc, dec = FederatedModel.build()
        self.autoencoder = tf.keras.models.Model(inputs=input_format, outputs=dec(enc(input_format)))
        self.autoencoder.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='mse', metrics=[])
        threading.Thread.__init__(self)

    def recv_single_sample(self):
        """Receive one data sample from data simulator

        :return: single data sample
        """
        conn, addr = self.data_socket.accept()
        with conn:
            while True:
                data = conn.recv(6024)
                if not data:
                    break
                conn.close()
                return json.loads(data)[0], json.loads(data)[1]  # convert data from binary json to list

    def run(self):
        """Collect samples until the data batch size is reached. Save this batch

        :return: -
        """
        while 1:
            try:
                # receive one sample and add it to container and store receiving time

                x, y = self.recv_single_sample()
                self.data_container.append(x)
                self.label_container.append(y)
                self.time_container.append(datetime.now())

            except Exception as e:
                logging.error("Error occured while data receiving!")
                print(e)

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
        prediction = ["ANOMALY" if l else "BENIGN" for l in outliers]
        labels_true = ["BENIGN" if k == "BENIGN" else "ANOMALY" for k in true_labels]

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



    def confusion_matrice(self, prediction:[], true_labels:[]):
        """Calculate confusion matrice

        :param prediction: Array containing the predicted labels
        :param true_labels: Array containing the true labels
        :return: False positives, True positives, False negatives, True negatives
        """
        fp = 0
        tp = 0
        fn = 0
        tn = 0
        for i in range(0, len(prediction)):
            if (prediction[i] == "ANOMALY" and true_labels[i] == "BENIGN"):
                fp += 1
            elif (prediction[i] == "ANOMALY" and true_labels[i] == "ANOMALY"):
                tp += 1
            elif (prediction[i] == "BENIGN" and true_labels[i] == "ANOMALY"):
                fn += 1
            elif (prediction[i] == "BENIGN" and true_labels[i] == "BENIGN"):
                tn += 1

        return fp, tp, fn, tn

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

    def calculate_metrics(self):
        """Calculate evaluation rates from last predictions from the saved confusion matrix

        :return: False positive rate, true positive rate, accuracy
        """
        fp = self.metrics[0]
        tp = self.metrics[1]
        fn = self.metrics[2]
        tn = self.metrics[3]
        fpr = (fp + tn) and fp / (fp + tn)
        tpr = (tp + fn) and tp / (tp + fn)
        ac = (tp + tn) / (tp + tn + fp + fn)
        self.metrics = [0, 0, 0, 0]
        return fpr, tpr, ac



class Client():
    """
    Class to create a single federated client. Parameters ID, Port and Receiving Port are received from arguments
    Optimized client for real-world dataset.
    Federated learning client that receives the global model weights, detects attacks in the arriving real-world
    datastreams from the data simulator, and trains the local model afterward.
    This model is transmitted to the Server for weight aggregation using FedAVG

    :args: ClientID ClientPort DataReceivingPort
    """

    def client_update(self, old_server_weights: [], dataset:[]):
        """
        Performs training using the received server model weights on the client's dataset

        :param dataset: current dataset for training in this round
        :return: new weights after training
        """
        input_format = keras.layers.Input(shape=(input_size,))
        enc, dec = FederatedModel.build()
        autoencoder = tf.keras.models.Model(inputs=input_format, outputs=dec(enc(input_format)))
        autoencoder.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='mse', metrics=[])
        autoencoder.set_weights(old_server_weights)

        history = autoencoder.fit(dataset, dataset, verbose=1, epochs=0, batch_size=32)

        weights = autoencoder.get_weights()  # get new weights
        return weights


    def recv_global_weights_from_server(self, s:socket):
        """
        Receive global weights from the server

        :return: latest global server weights
        """
        binary_data = b''
        s.settimeout(300.0)  # after 5min, assume the server is down
        conn, addr = s.accept()
        with conn:
            while True:
                recv_data = conn.recv(30000)
                if not recv_data:
                    break
                binary_data = b"".join([binary_data, recv_data])
        conn.close()

        received_data = json.loads(binary_data)
        new_global_weights = self.unpack_weights(received_data)
        return new_global_weights



    def unpack_weights(self, received_data:bytearray):
        """Unpack local weigths from json format

        :param received_data: Bytedata received from server
        :return: List of latest server weights
        """
        weight_list = []
        for layer in range(0, len(received_data)):
            weight_list.append(np.array(received_data[layer]))
        return weight_list



    def pack_weights(self, global_weights: [], metrics:[]):
        """Pack local weigths into json format for transmission

        :param global_weights: New weights calculated on this client
        :param metrics: Metrics that are transmitted to server
        :return: Converted and encrypted message
        """
        list_of_weights = [f'client_{CLIENT_ID}', [metrics[0]], [metrics[1]], [metrics[2]], [metrics[3]]]
        for layer in range(0, len(global_weights)):
            list_of_weights.append(global_weights[layer].tolist())
        return json.dumps(list_of_weights).encode()


    def send_weights_to_server(self, weights: [], metrics:[]):
        """Send local weights back to server

        :param weights:
        :param metrics:
        :return:
        """
        bytes_weights_data = self.pack_weights(weights, metrics)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, SERVER_PORT))
            s.sendall(bytes_weights_data)


    def register_client(self, ID:int, Client_Port:int, Server_port:int):
        """Register client at the server

        :param ID: Clients ID
        :param Client_Port: Port of this client
        :param Server_port: Port of the central server
        :return: True if successful
        """
        logging.info("Try registration: ")
        bytes_weights_data = json.dumps(
            ["registration", ID, Client_Port, Server_port]).encode()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, SERVER_PORT))
            s.sendall(bytes_weights_data)
        return True


    def no_data_message(self, ID:int, Client_Port:int, Server_port:int):
        """Send message with no data flag to the server

        :param ID: Clients ID
        :param Client_Port: Port of this client
        :param Server_port: Port of the central server
        :return: True if successful
        """
        bytes_weights_data = json.dumps(
            ["no data", ID, Client_Port, Server_port]).encode()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, SERVER_PORT))
            s.sendall(bytes_weights_data)
        return


    def update_registration(self, registered:bool, last_registration:datetime):
        """Check if registration is still valid.

        :param registered:
        :param last_registation:
        :return: True if client is registered, False if not and the time of last successful registration
        """

        if (not registered) and ((datetime.now() - last_registration).total_seconds() > 15):
            try:
                logging.info("Registration request")
                # try registration and update time
                last_registration = datetime.now()
                registered = self.register_client(CLIENT_ID, CLIENT_PORT, SERVER_PORT)
                if registered:
                    logging.info(f"Registration successful! {CLIENT_ID} {CLIENT_PORT} {DATA_RECEIVING_PORT}")
                    return True, last_registration
            except Exception as e:
                logging.warning(f"Client {CLIENT_ID}: Failed to register at server. Retry in 10sec.")
                return False, last_registration
        else:
            return True, last_registration


if __name__ == "__main__":
    logging.info(f"START CLIENT {sys.argv[1]} on {sys.argv[2]} with datasource on {sys.argv[3]}")

    CLIENT_ID = int(sys.argv[1])
    CLIENT_PORT = int(sys.argv[2])
    DATA_RECEIVING_PORT = int(sys.argv[3])


    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # open socket
    s.bind((HOST, CLIENT_PORT))  # bind process to port
    s.listen()

    # initialize variables
    registered = False
    train_count = 0
    last_registration = datetime.now() - timedelta(seconds=90)

    # create data receiving thread
    data_thread = Data_Receiving_Thread()
    data_thread.start()
    Client = Client()

    while 1:
        registered, last_registration = Client.update_registration(registered, last_registration)

        if registered:
            try:
                GLOBAL_WEIGHTS = Client.recv_global_weights_from_server(s)
            except socket.timeout:
                registered = False
                logging.warning("Server not available")
                continue

            if len(data_thread.train_data_batch) >= 1:

                # get all traindata from data thread
                train_dataset = np.array([item for sublist in data_thread.train_data_batch for item in
                                          sublist])
                logging.info(f"Start training on client {CLIENT_ID} with {len(train_dataset)} samples")

                client_weights = Client.client_update(GLOBAL_WEIGHTS, train_dataset)

                # add 1 to traincount to skip prediction in the first rounds
                train_count += 1
                data_thread.train_data_label = []
                data_thread.train_data_batch = []

                try:
                    Client.send_weights_to_server(client_weights, data_thread.metrics)
                    fpr, tpr, ac = data_thread.calculate_metrics()
                    with open(f'results/clientmetrics/client_{CLIENT_ID}tpr_fpr.txt', 'a') as file:
                        file.write(f'{tpr}		{fpr}	{ac}\n')
                except:
                    logging.error("Could not send weights back")

                logging.info("Finished training, closed session")
                K.clear_session()  # clear model
            else:
                logging.warning("No data available for training")
                Client.no_data_message(CLIENT_ID, CLIENT_PORT, SERVER_PORT)