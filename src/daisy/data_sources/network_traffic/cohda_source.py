# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Implementations of the data source processor interface that allows the processing
and provisioning of pyshark packets that are captured from cohda boxes.

Author: Seraphin Zunzer, Fabian Hofmann
Modified: 19.04.24
"""

from datetime import datetime
from typing import Callable

import numpy as np

from ...data_sources.data_processor import SimpleDataProcessor
from ...data_sources.network_traffic.pyshark_processor import (
    default_f_features,
    default_nn_aggregator,
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
        name: str = "",
        f_features: tuple[str, ...] = default_f_features,
        nn_aggregator: Callable[[str, object], object] = default_nn_aggregator,
    ):
        """Creates a new cohda processor for a specific client.

        :param client_id: ID of client.
        :param events: List of labeled, self-descriptive, events by which one can
        label individual data points with.
        :param name: Name of processor for logging purposes.
        :param f_features: Selection of features that every data point will have
        after processing.
        :param nn_aggregator: List aggregator that is able to aggregator dictionary
        values that are lists into singleton values, depending on the key they are
        sorted under.
        """
        super().__init__(
            pyshark_map_fn(),
            pyshark_filter_fn(f_features),
            pyshark_reduce_fn(nn_aggregator),
            name,
        )

        self._client_id = client_id
        self._events = events

    def reduce(self, d_point: dict) -> np.ndarray:
        """Transform the pyshark data point directly into a numpy array after also
        adding the true label to the observation based on the provided (labeled) events.

        :param d_point: Data point as dictionary.
        :return: Labeled data point as vector.
        """
        d_point["label"] = 0
        for event in self._events:
            client, (start_time, end_time), protocols, addresses, label = event
            if (
                client == self._client_id
                and start_time
                <= datetime.strptime(d_point["meta.time"], "%Y-%m-%d %H:%M:%S.%f")
                <= end_time
                and any([x in d_point["meta.protocols"] for x in protocols])
                and all([x in d_point["ip.addr"] for x in addresses])
            ):
                d_point["label"] = label
                break
        return super().reduce(d_point)


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
