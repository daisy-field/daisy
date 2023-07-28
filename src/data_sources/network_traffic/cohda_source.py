"""
    Implementations of the data source processor interface that allows the processing and provisioning of pyshark
    packets that are captured from cohda boxes.

    Author: Seraphin Zunzer, Fabian Hofmann
    Modified: 14.06.23
"""

from datetime import datetime

import numpy as np

from src.data_sources.network_traffic.pyshark_source import PysharkProcessor, default_f


# TODO factory functions

class CohdaProcessor(PysharkProcessor):
    """An extension of the pyshark processor to support the labeling of the data stream for evaluation purposes. Labels
    are appended according to the used protocol, timestamps, source and destination ip addresses.
    """
    _client_id: int
    _events: list[tuple[int, tuple[datetime, datetime], list[str], list[str], str]]

    def __init__(self, client_id: int, events: list[tuple[int, tuple[datetime, datetime], list[str], list[str], str]],
                 f_features: tuple[str, ...] = default_f, ):
        """Creates a new cohda processor for a specific client.

        :param client_id: ID of client that
        :param f_features: Selection of features that every data point will have after processing.
        :param events: List of labeled, self-descriptive, events by which one can label individual data points with.
        """
        self._client_id = client_id
        self._events = events
        super().__init__(f_features)

    def reduce(self, d_point: dict) -> np.ndarray:
        """Transform the pyshark data point directly into a numpy array after also adding the true label to the
        observation based on the provided (labeled) events.

        :param d_point: Data point as dictionary.
        :return: Labeled data point as vector.
        """
        for event in self._events:
            client, (start_time, end_time), protocols, addresses, label = event
            if client == self._client_id and start_time <= d_point['meta.time'].to_datetime() <= end_time and \
                    any([x in d_point['meta.protocols'] for x in protocols]) and \
                    all([x in d_point['ip.addr'] for x in addresses]):
                d_point['label'] = label
                break
        if "label" in d_point:
            d_point['label'] = "Normal"

        return super().reduce(d_point)


# Existing datasets captured on Cohda boxes 2 and 5 on March 6th contains attacks in the following:
march23_events: list[tuple[int, tuple[datetime, datetime], list[str], list[str], str]] = \
    [(5, (datetime(2023, 3, 6, 12, 34, 17), datetime(2023, 3, 6, 12, 40, 28)),
      ["http", "tcp"], ["192.168.213.86", "185."], "Installation Attack Tool"),
     (5, (datetime(2023, 3, 6, 12, 49, 4), datetime(2023, 3, 6, 13, 23, 16)),
      ["ssh", "tcp"], ["192.168.230.3", "192.168.213.86"], "SSH Brute Force"),
     (5, (datetime(2023, 3, 6, 13, 25, 27), datetime(2023, 3, 6, 13, 31, 11)),
      ["ssh", "tcp"], ["192.168.230.3", "192.168.213.86"], "SSH Privilege Escalation"),
     (2, (datetime(2023, 3, 6, 12, 49, 4), datetime(2023, 3, 6, 13, 23, 16)),
      ["ssh", "tcp"], ["192.168.230.3", "130.149.98.119"], "SSH Brute Force Response"),
     (2, (datetime(2023, 3, 6, 13, 25, 27), datetime(2023, 3, 6, 13, 31, 11)),
      ["ssh", "tcp"], ["192.168.230.3", "130.149.98.119"], "SSH Data Leakage")]
