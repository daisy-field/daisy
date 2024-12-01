# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Content used for the Dataset Demo from March 6th 2023.

Author: Jonathan Ackerschewski, Fabian Hofmann
Modified: 04.11.2024
"""

import ipaddress

import sys


def csv_nn_aggregator(key: str, value: object) -> int:
    """Simple, exemplary value aggregator. Takes a non-numerical (i.e. string) key-value
    pair and attempts to converted it into an integer / float. This example does not
    take the key into account, but only checks the types of the value to proceed. Note,
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
        except ipaddress.AddressValueError:
            pass
        try:
            return int(ipaddress.IPv6Address(value)) % sys.maxsize
        except ipaddress.AddressValueError:
            pass
        try:
            return int(value, 16)
        except ValueError:
            pass
        return hash(value)
    raise ValueError(f"Unable to aggregate non-numerical item: {key, value}")


def cic_label_data_point(d_point: dict) -> dict:
    """removes the original label key and labels the data points coorectly.

    :param client_id: Client ID.
    :param d_point: Data point as dictionary.
    :return: Labeled data point.
    """
    label = d_point.pop(" Label")
    if label == "BENIGN":
        d_point["label"] = 0
    else:
        d_point["label"] = 1

    return d_point  # np.asarray(l_point)
