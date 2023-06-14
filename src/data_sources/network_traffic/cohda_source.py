"""
    Implementations of the data source processor interface that allows the processing and provisioning of pyshark
    packets that are captured from cohda boxes.

    Author: Seraphin Zunzer, Fabian Hofmann
    Modified: 14.06.23
"""

from datetime import datetime
from typing import Tuple


import numpy as np

from src.data_sources.network_traffic.pyshark_source import PysharkProcessor, default_f

class CohdaProcessor(PysharkProcessor):
    """An extension of the pyshark processor to support the labeling of the data stream for evaluation purposes.
    Currently static, as the existing datasets captured on Cohda boxes 2 and 5 on March 6th contain attacks. Labels are
    appended according to the used protocol, timestamps, source and destination ip addresses.
    """
    _client_id: int

    def __init__(self, client_id: int, f_features: tuple[str, ...] = default_f):
        """Creates a new cohda processor for a specific client.

        :param client_id: ID of client that
        :param f_features: Selection of features that every data point will have after processing.
        """
        self._client_id = client_id
        super().__init__(f_features)

    def filter_anomaly(self, client_id: int, client_target: int, attack_time: Tuple[datetime, datetime],
                       target_protocols: [],
                       target_ip_addresses: [], label: string, data_point: []):
        """Check if datapoint is an anomaly and add label if condition is fulfilled
        :param client_id: Client ID of the client this cohda sorce sends the traffic to
        :param client_target: The ID of the client that was affected of this attack
        :param target_protocols: Protocols that were targeted
        :param start_time: Start time of the attack
        :param end_time: End time of the attack
        :param target_ip_addresses: List of IP addresses that were targeted
        :param label: String that should be the label
        :param data_point: Datapoint to test for anomaly
        :return: dict: data_point with appended label if anomaly conditions are fulfilled
        """

        if data_point.get("label") is None and \
                (client_id == client_target and
                 _in_interval(data_point['meta.time'].to_datetime(), attack_time) and
                 any([x in datapoint['meta.protocols'] for x in target_protocols]) and
                 all([x in o_point['ip.addr'] for x in target_ip_addresses])):
            data_point['label'] = label
            return data_point

    def reduce(self, d_point: dict) -> np.ndarray:
        """Transform the pyshark data point directly into a numpy array after also adding the true label to the
        observation. All of this is hardcoded right now, as there is no additional labeled pcap dataset to be used from
        the Cohda boxes.

        :param d_point: Data point as dictionary.
        :return: Labeled data point as vector.
        """


        attack_tool_ts = Tuple[datetime(2023, 3, 6, 12, 34, 00), datetime(2023, 3, 6, 12, 40, 00)]
        brute_force_ts = Tuple[datetime(2023, 3, 6, 12, 49, 00), datetime(2023, 3, 6, 13, 23, 00)]
        privilege_ts = Tuple[datetime(2023, 3, 6, 13, 25, 00), datetime(2023, 3, 6, 13, 32, 00)]

        self.label_anomalies(CLIENT_ID, 5, attack_tool_ts,
                         ["http", "tcp"], ["192.168.213.86", "185."], "Installation Attack Tool", d_point)
        self.label_anomalies(CLIENT_ID, 5, brute_force_ts,
                         ["ssh", "tcp"], ["192.168.230.3", "192.168.213.86"], "SSH Brute Force", d_point)
        self.label_anomalies(CLIENT_ID, 5, privilege_ts,
                         ["ssh", "tcp"], ["192.168.230.3", "192.168.213.86"], "SSH Privilege Escalation", d_point)

        self.label_anomalies(CLIENT_ID, 2, brute_force_ts,
                     ["ssh", "tcp"], ["192.168.230.3", "130.149.98.119"], "SSH Brute Force Response", d_point)
        self.label_anomalies(CLIENT_ID, 2, privilege_ts,
                     ["ssh", "tcp"], ["192.168.230.3", "130.149.98.119"], "SSH Data Leakage", d_point)

        if d_point.get("label") is None:
            d_point['label'] = "Normal"

        return super().reduce(d_point)


def _in_interval(timestamp: datetime, interval: [datetime, datetime]) -> bool:
    """Checks if a timestamp is in an interval.

    :param timestamp: Timestamp to check.
    :param interval: Array of two datetime objects defining interval (start, end).
    :return: True if timestamp time is in time interval.
    """
    return interval[0] <= timestamp <= interval[1]



