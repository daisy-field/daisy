# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Implementations of the data source processor interface that allows the processing
and provisioning of pyshark packets that are captured from cohda boxes.

Author: Fabian Hofmann, Seraphin Zunzer
Modified: 07.06.24
"""

from datetime import datetime
from typing import Callable

import numpy as np

from ...data_sources.data_processor import SimpleDataProcessor
from ...data_sources.network_traffic.pyshark_processor import (
    pyshark_map_fn,
    pyshark_filter_fn,
    pyshark_reduce_fn,
)


class CohdaProcessor(SimpleDataProcessor):
    """An extension of the pyshark processor to support the labeling of the data
    stream for evaluation purposes. Labels are appended according to the used
    protocol, timestamps, source and destination ip addresses.
    """

    _client_id: int
    _events: list[tuple[int, tuple[datetime, datetime], list[str], list[str], int]]

    def __init__(
        self,
        client_id: int,
        events: list[tuple[int, tuple[datetime, datetime], list[str], list[str], int]],
        map_fn: Callable[[object], dict] = pyshark_map_fn(),
        filter_fn: Callable[[dict], dict] = pyshark_filter_fn(),
        reduce_fn: Callable[[dict], np.ndarray | dict] = pyshark_reduce_fn(),
        name: str = "",
    ):
        """Creates a new cohda processor for a specific client.

        :param client_id: ID of client.
        :param events: List of labeled, self-descriptive, events by which one can
        label individual data points with.
        :param map_fn: Map function, which receives a data point and maps it to a
        dictionary.
        :param filter_fn: Filter function, which receives the map output and
        filters/selects its features.
        :param reduce_fn: Reduce function, which receives the filter output and
        transforms into a numpy array (if serving as input for ML tasks), to be executed
        after the pyshark data point (packet) has been labeled.
        :param name: Name of processor for logging purposes.
        """
        super().__init__(
            map_fn=map_fn,
            filter_fn=filter_fn,
            reduce_fn=reduce_fn,
            name=name,
        )

        self._client_id = client_id
        self._events = events

    def reduce(self, d_point: dict) -> np.ndarray:
        """Transform the pyshark data point directly into a numpy array after also
        adding the true label to the observation based on the provided (labeled) events.

        :param d_point: Data point as dictionary.
        :return: Labeled data point as vector.
        """
        d_point = cohda_label_packets(d_point, self._events, self._client_id)
        return super().reduce(d_point)


def cohda_label_packets(
    d_point: dict,
    events: list[tuple[int, tuple[datetime, datetime], list[str], list[str], int]],
    client_id: int,
) -> dict:
    """Labels the pyshark data points based on the provided (labeled) events.

    :param d_point: Data point as dictionary.
    :param events: List of labeled, self-descriptive, events by which one can
    label individual data points with.
    :param client_id: ID of client.
    :return: Labeled data point.
    """
    d_point["label"] = 0
    for event in events:
        client, (start_time, end_time), protocols, addresses, label = event
        if (
            client == client_id
            and start_time
            <= datetime.strptime(d_point["meta.time"], "%Y-%m-%d %H:%M:%S.%f")
            <= end_time
            and any([x in d_point["meta.protocols"] for x in protocols])
            and d_point["ip.addr"] is not np.nan
            and all([x in d_point["ip.addr"] for x in addresses])
        ):
            d_point["label"] = label
            break
    return d_point


# Existing datasets captured on Cohda boxes 2 and 5 on March 6th (2023)
# contains attacks in the following:
# 1: "Installation Attack Tool"
# 2: "SSH Brute Force"
# 3: "SSH Privilege Escalation"
# 4: "SSH Brute Force Response"
# 5: "SSH Data Leakage"
march23_events: list[
    tuple[int, tuple[datetime, datetime], list[str], list[str], int]
] = [
    (
        5,
        (datetime(2023, 3, 6, 12, 34, 17), datetime(2023, 3, 6, 12, 40, 28)),
        ["http", "tcp"],
        ["192.168.213.86", "185."],
        1,
    ),
    (
        5,
        (datetime(2023, 3, 6, 12, 49, 4), datetime(2023, 3, 6, 13, 23, 16)),
        ["ssh", "tcp"],
        ["192.168.230.3", "192.168.213.86"],
        2,
    ),
    (
        5,
        (datetime(2023, 3, 6, 13, 25, 27), datetime(2023, 3, 6, 13, 31, 11)),
        ["ssh", "tcp"],
        ["192.168.230.3", "192.168.213.86"],
        3,
    ),
    (
        2,
        (datetime(2023, 3, 6, 12, 49, 4), datetime(2023, 3, 6, 13, 23, 16)),
        ["ssh", "tcp"],
        ["192.168.230.3", "130.149.98.119"],
        4,
    ),
    (
        2,
        (datetime(2023, 3, 6, 13, 25, 27), datetime(2023, 3, 6, 13, 31, 11)),
        ["ssh", "tcp"],
        ["192.168.230.3", "130.149.98.119"],
        5,
    ),
]
