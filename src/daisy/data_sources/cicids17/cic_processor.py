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

import ipaddress
from ipaddress import AddressValueError

import numpy as np
import sys

from ...data_sources.data_processor import SimpleDataProcessor


class CICProcessor(SimpleDataProcessor):
    """An extension of the simple processor to add the labeling of the data"""

    def __init__(
        self,
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
            name=name,
        )

    def default_nn_aggregator(self, key: str, value: object) -> int:
        """Simple, exemplary value aggregator. Takes a non-numerical (i.e. string) key-value
        pair and attempts to converted it into an integer. This example does not take
        the key into account, but only checks the types of the value to proceed. Note,
        that ipv6 are lazily converted to 32 bit (collisions may occur).

        :param key: Name of pair, which always a string.
        :param value: Arbitrary non-numerical value to be converted.
        :return: Converted numerical value.
        :raises ValueError: If value cannot be converted.
        """
        if isinstance(value, list):
            value.sort()
            return hash(str(value))

        if isinstance(value, str):
            try:
                return int(ipaddress.IPv4Address(value))
            except AddressValueError:
                pass
            try:
                return int(ipaddress.IPv6Address(value)) % sys.maxsize
            except AddressValueError:
                pass
            try:
                return int(value, 16)
            except ValueError:
                pass
            return hash(value)

        raise ValueError(f"Unable to aggregate non-numerical item: {key, value}")

    def reduce(self, d_point: dict) -> np.ndarray:
        """Transform the pyshark data point directly into a numpy array after also
        adding the true label to the observation based on the provided (labeled) events.

        :param d_point: Data point as dictionary.
        :return: Labeled data point as vector.
        """
        label = d_point.pop(" Label")
        if label == "BENIGN":
            d_point["label"] = 0
        else:
            d_point["label"] = 1

        l_point = []
        for key, value in d_point.items():
            if not isinstance(value, int | float):
                value = self.default_nn_aggregator(key, value)
            try:
                if np.isnan(value):
                    value = 0
            except TypeError as e:
                raise ValueError(f"Invalid k/v pair: {key}, {value}") from e
            l_point.append(value)

        self._logger.warning(l_point)

        return np.asarray(l_point)
