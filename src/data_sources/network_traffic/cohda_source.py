"""
    TODO

    Author: Seraphin Zunzer, Fabian Hofmann
    Modified: 02.05.23
"""

from datetime import datetime

import numpy as np

from src.data_sources.network_traffic.pyshark_source import PysharkProcessor, default_f

attack_tool_ts = [datetime(2023, 3, 6, 12, 34, 00), datetime(2023, 3, 6, 12, 40, 00)]
brute_force_ts = [datetime(2023, 3, 6, 12, 49, 00), datetime(2023, 3, 6, 13, 23, 00)]
privilege_ts = [datetime(2023, 3, 6, 13, 25, 00), datetime(2023, 3, 6, 13, 32, 00)]


class CohdaProcessor(PysharkProcessor):
    """An extension of the pyshark processor to support the labeling of the data stream for evaluation purposes.
    Currently static, as the existing datasets captured on Cohda boxes 2 and 5 on March 6th contain attacks. Labels are
    appended according to the used protocol, timestamps, source and destination ip addresses.
    """
    _client_id: int

    def __init__(self, client_id: int, f_features: tuple[str, ...] = default_f):
        """Creates a new cohda . FIXME

        :param client_id: ID of client that
        :param f_features: Selection of features that every data point will have after processing.
        """
        self._client_id = client_id
        super().__init__(f_features)

    def reduce(self, d_point: dict) -> np.ndarray:
        """Transform the pyshark data point directly into a numpy array after also adding the true label to the
        observation. All of this is hardcoded right now, as there is no additional labeled pcap dataset to be used from
        the Cohda boxes.

        :param d_point: Data point as dictionary.
        :return: Labeled data point as vector.
        """
        if (self._client_id == 5 and _is_interval(d_point['meta.time'].to_datetime(), attack_tool_ts)
                and any([x in d_point['meta.protocols'] for x in ["http", "tcp"]])
                and all([x in d_point['ip.addr'] for x in ["192.168.213.86", "185."]])):
            d_point['label'] = "Installation Attack Tool"
        elif (self._client_id == 5 and _is_interval(d_point['meta.time'].to_datetime(), brute_force_ts)
              and any([x in d_point['meta.protocols'] for x in ["ssh", "tcp"]])
              and all([x in d_point['ip.addr'] for x in ["192.168.230.3", "192.168.213.86"]])):
            d_point['label'] = "SSH Brute Force"
        elif (self._client_id == 5 and _is_interval(d_point['meta.time'].to_datetime(), privilege_ts)
              and any([x in d_point['meta.protocols'] for x in ["ssh", "tcp"]])
              and all([x in d_point['ip.addr'] for x in ["192.168.230.3", "192.168.213.86"]])):
            d_point['label'] = "SSH Privilege Escalation"
        elif (self._client_id == 2 and _is_interval(d_point['meta.time'].to_datetime(), brute_force_ts)
              and any([x in d_point['meta.protocols'] for x in ["ssh", "tcp"]])
              and all([x in d_point['ip.addr'] for x in ["192.168.230.3", "130.149.98.119"]])):
            d_point['label'] = "SSH Brute Force Response"
        elif (self._client_id == 2 and _is_interval(d_point['meta.time'].to_datetime(), privilege_ts)
              and any([x in d_point['meta.protocols'] for x in ["ssh", "tcp"]])
              and all([x in d_point['ip.addr'] for x in ["192.168.230.3", "130.149.98.119"]])):
            d_point['label'] = "SSH Data Leakage"
        else:
            d_point['label'] = "Normal"

        return super().reduce(d_point)


def _is_interval(timestamp: datetime, interval: [datetime, datetime]) -> bool:
    """Checks if a timestamp is in an interval.

    :param timestamp: Timestamp to check.
    :param interval: Array of two datetime objects defining interval (start, end).
    :return: True if timestamp time is in time interval.
    """
    return interval[0] <= timestamp <= interval[1]
