"""
    Collection of functions to process detected anomalies.
        process_anomalies: write all detected anomalies together with true label in file
        store_anomalies: write detected anomalies to file based on anomaly type

    Author: Seraphin Zunzer
    Modified: 09.08.23
"""
from datetime import datetime
from typing import Tuple


# TODO must be streamlined, made more generic for arbitrary labels and attacks

def process_anomalies(addr: Tuple[str, int], predictions: [], true_labels: []):
    """Function to process anomalies, e.g. delete packets, throw alerts etc.
    In this case write anomaly to file with timestamp.

    :param predictions:
    :param true_labels:
    :return:
    """
    now = datetime.now()
    timestamp = now.strftime("%m/%d/%Y, %H:%M:%S")

    with open(f"results_{addr}.txt", "a") as txt_file:
        for i in range(len(predictions)):
            if predictions[i] == "anomaly":
                txt_file.write(f" {timestamp} - {true_labels[i]}  \n")


def store_anomalies(true_labels: [], z_scores: [], time_needed: [], outliers: []):
    """Store times needed for processing, the result of classification and the calculated score in specific txt file

    :param true_labels: True labels list
    :param z_scores: Z_scores list
    :param time_needed: Time needed for detection
    :param outliers: True/False list if datasample is classified as anomaly
    :return: Write detection time and true label to given file
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
