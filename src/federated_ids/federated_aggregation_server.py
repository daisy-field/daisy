#
#   Created: 2023
#   Author: Seraphin Zunzer
#   License: MIT License
# ---------------------------------------------------------------


# Optimized server for real-world dataset.
# Federated learning server which starts the federated learning process on the clients,
# receives the newly trained client models,
# and aggregates them using FedAVG.


import numpy as np
import keras
import tensorflow as tf
import json
import socket
from tabulate import tabulate
from tensorflow.keras.models import Sequential
from src.federated_ids import federated_model as fm

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
SERVER_PORT = 65432  # Port to listen on (non-privileged ports are > 1023)
client_dict = {}
new_registrations = {}
encryption = True
ROUNDS = 1000
DROPOUT = 0.5
DATASET_SIZE = 70


# list with all clients that are offline
OFFLINE_CLIENTS = []

# list of all active clients
ACTIVE_CLIENTS = []

# list containing the latest client states
client_states= []


def start_client_training(client_port, server_weights_data):
    """
    Connect to client, send global weights which start the training

    :param client_port: Port of the client to send the data to
    :param server_weights_data: List of the latest server model weights
    :return: -
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, client_port))
        s.sendall(aes_encrypt(server_weights_data))


def receive_client_message(s):
    """
    Receive updated model weights

    :param s: Socket to receive data on
    :return: Multiple variables, none if it is a no data or registration message, the client data and metrics otherwise
    """
    while 1:
        s.settimeout(60.0)
        binary_data = b''
        conn, addr = s.accept()
        with conn:
            while True:
                recv_data = conn.recv(30000)
                if not recv_data:
                    break
                binary_data = b"".join([binary_data, recv_data])
            conn.close()
        received_data = json.loads(aes_decrypt(binary_data))
        if received_data[0] == "registration":
            process_registration(received_data[1], received_data[2])
            return None, None, None, None, None, None
        elif received_data[0] =="no data":
            process_no_data(received_data[1], received_data[2])
            return None, None, None, None, None, None
        else:
            answering_client = received_data[0]  # pay attention to the client id that comes with the weights
            fp = received_data[1]
            tp =received_data[2]
            fn =received_data[3]
            tn = received_data[4]
            client_weights = unpack_weights(received_data)
            return answering_client, client_weights, fp[0], tp[0], fn[0], tn[0]


def unpack_weights(received_data):
    """
    Unpack clients weights from json format
    
    :param received_data: 
    :return: 
    """
    weight_list = []
    for layer in range(5, len(received_data)):
        weight_list.append(np.array(received_data[layer]))
    return weight_list


def pack_weights(global_weights):
    """pack global weigths into json format for transmission"""
    list_of_weights = []
    for layer in range(0, len(global_weights)):
        list_of_weights.append(global_weights[layer].tolist())
    return json.dumps(list_of_weights).encode()


def recv_registrations(s):
    """
    Receive registration messages on startup

    :param s:
    :return:
    """
    while 1:
        s.settimeout(3.0)
        binary_data = b''
        conn, addr = s.accept()
        with conn:
            while True:
                recv_data = conn.recv(10000)
                if not recv_data:
                    break
                binary_data = b"".join([binary_data, recv_data])
            conn.close()
            data = json.loads(aes_decrypt(binary_data))

            # check if flags are set
            if str(data[0]) == "registration":
                process_registration(data[1], data[2])
            if str(data[0]) == "no data":
                process_no_data(data[1], data[2])


def process_registration(new_client_id, new_client_port):
    """
    Process a new registration

    :param new_client_id:
    :param new_client_port:
    :return:
    """
    print(f"New Client registration: {new_client_id}, {new_client_port}")
    new_registrations["client_" + str(new_client_id)] = int(new_client_port)
    client_states.append(["client_" + str(new_client_id), int(new_client_port), "new registration", "-", "-", "-", "-"])

def process_no_data(client_id, client_port):
    """
    Process no data messages
    
    :param client_id: 
    :param client_port: 
    :return: 
    """
    client_states.append(["client_" + str(client_id), int(client_port), "no data", "-", "-", "-", "-"])


def averaged_sum(global_weights, client_weights):
    """
    Calculate fedavg between global weigths and new weigths

    :param global_weights:
    :param client_weights:
    :return:
    """
    try:
        for i in range(0, len(global_weights)):
            for j in range(0, len(global_weights[i])):
                global_weights[i][j] = (global_weights[i][j] + client_weights[i][j]) / 2
            # if list index out of range: check if model on client and server is equal
    except:
        print("Malformed weights received!")
    return global_weights



# -----------------MAIN LOOP------------------


if __name__ == "__main__":

    # initialize global model
    input_format = keras.layers.Input(shape=(DATASET_SIZE,))
    enc, dec = fm.FederatedModel().build()
    global_model = tf.keras.models.Model(inputs=input_format, outputs=dec(enc(input_format)))

    # open server socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, SERVER_PORT))
    s.listen()

    # start global training loop
    for round in range(ROUNDS):
        print('_' * 50)
        print(f"START ROUND {round} ")

        # get the global model's weights, used as initial weights for all local models
        global_weights = global_model.get_weights()

        changed_weights = False  # variable if at least one client is available and any weights change
        client_states = []  # list to save client states

        try:
            recv_registrations(s)  # check if there are any new registrations
        except socket.timeout:
            pass

        # loop through each client and start training
        for id in client_dict.keys():
            port = client_dict[id]

            try:  # start training on client
                bytes_weights_data = pack_weights(global_weights)
                start_client_training(port, bytes_weights_data)  # start training
            except Exception as e:
                print(e)
                client_states.append([id, client_dict[id], "down"])  # if client is not available: continue
                pass

        print("Started training on all available clients.\n")

       # as long as not all clients responded
        while not all(elem in [k[0] for k in client_states] for elem in client_dict.keys()):  # receive training results
            try:
                # try to receive messages
                answering_client, local_client_weights = receive_client_message(s)

                # if a client returned weights
                if not (answering_client is None or local_client_weights is None ):
                    print("Started averaging - ", end="")

                    #average weights
                    average_weights = averaged_sum(global_weights, local_client_weights)

                    # update global model
                    global_model.set_weights(average_weights)
                    changed_weights = True

                    print(f"{answering_client} finished!")
                    client_states.append([answering_client, client_dict[answering_client], "up", fp, tp, fn, tn])

                    OFFLINE_CLIENTS[:] = (value for value in OFFLINE_CLIENTS if value != answering_client)          #remove element if in offline list
                else:
                    #skip if malformed weights received
                    print("")

            # no more responses from any clients
            except socket.timeout:
                break

        # analyze which and how many clients have responded
        active = 0
        remove =[]
        for i in client_dict.keys():  # analyse which clients responded
            found = False
            for j in client_states:
                if j[0] == i:
                    if (j[2] == "up"):
                        active += 1
                    found = True
            if found == False:
                client_states.append([i, client_dict[i], "no response -> deleted"])
                OFFLINE_CLIENTS.append(i)
                if OFFLINE_CLIENTS.count(i) >= 5:    #throw out not responding clients
                    remove.append(i)

        for i in remove:
            del client_dict[i]

        print(tabulate(sorted(client_states, key=lambda x: x[1]), headers=['Client', 'Port', 'State', 'False-Positives', 'True-positives', 'False-negatives', 'True-negatives']),
              "\n")  # show client table with states
        print(f"Evaluation: TPR:{overall_tpr}, FPR: {overall_fpr}, AC: {overall_ac}, F1: {overall_f1}")
        client_dict = {**client_dict, **new_registrations}  # merge new registered clients into client dictionary
        new_registrations = {}
