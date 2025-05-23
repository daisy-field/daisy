# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Implementation of the data processor for supporting processing steps used for pyshark
packets, i.e. a pre-packaged extension of the data processor base class for ease of use.

Author: Jonathan Ackerschewski, Fabian Hofmann
Modified: 04.11.2024
"""

import ipaddress
import json
import logging
import sys
from collections import defaultdict
from ipaddress import AddressValueError
from typing import Callable, Self

import numpy as np
from pyshark.packet.fields import LayerField, LayerFieldsContainer
from pyshark.packet.layers.json_layer import JsonLayer
from pyshark.packet.layers.xml_layer import XmlLayer
from pyshark.packet.packet import Packet
from typing_extensions import deprecated

from ..data_processor import DataProcessor, flatten_dict

# Exemplary network feature filter, supporting cohda-box (V2x) messages, besides
# TCP/IP and ETH.
pcap_f_features = (
    "data.len",
    "eth.addr",
    "eth.dst",
    "eth.len",
    "eth.src",
    "ip.addr",
    "ip.dst",
    "ip.flags.df",
    "ip.flags.mf",
    "ip.flags.rb",
    "ip.src",
    "ipv6.addr",
    "ipv6.dst",
    "ipv6.src",
    "ipv6.tclass",
    "llc.control.ftype",
    "llc.control.n_r",
    "llc.control.n_s",
    "llc.dsap.ig",
    "llc.dsap.sap",
    "llc.ssap.cr",
    "llc.ssap.sap",
    "meta.len",
    "meta.number",
    "meta.protocols",
    "meta.time",
    "meta.time_epoch",
    "sll.eth",
    "sll.etype",
    "sll.halen",
    "sll.hatype",
    "sll.ltype",
    "sll.padding",
    "sll.pkttype",
    "sll.trailer",
    "sll.unused",
    "ssh.direction",
    "ssh.protocol",
    "tcp.dstport",
    "tcp.port",
    "tcp.segments.count",
    "tcp.srcport",
    "udp.port",
)


def pcap_nn_aggregator(key: str, value: object) -> int | float:
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
        except AddressValueError:
            pass
        try:
            return int(ipaddress.IPv6Address(value)) % sys.maxsize
        except AddressValueError:
            pass
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return int(value, 16)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return hash(value)

    raise ValueError(f"Unable to aggregate non-numerical item: {key, value}")


class PysharkProcessor(DataProcessor):
    """Extension of the data processor base class with pre-built processing steps
    specifically for pyshark packets.
    """

    def packet_to_dict(self) -> Self:
        """Adds a function to the processor that takes a data point which is a
        pyshark packet and converts it into a dictionary.
        """

        # noinspection DuplicatedCode
        def packet_to_dict_func(p: Packet) -> dict:
            p_dict = {}
            meta_dict = {
                "number": p.number,
                "len": p.length,
                "protocols": [x.layer_name for x in p.layers],
                "time_epoch": p.sniff_timestamp,
                "time": str(p.sniff_time),
            }
            p_dict.update({"meta": meta_dict})

            for layer in p.layers:
                p_dict.update(_add_layer_to_dict(layer))
            return p_dict

        return (
            self.add_func(lambda d_point: packet_to_dict_func(d_point))
            .flatten_dict()
            .ensure_features_exist()
        )

    def ensure_features_exist(self) -> Self:
        """Adds a function ensuring the following features exist in the result
        dictionary, if related features exist:

        sll.eth
        ipv6.addr
        ipv6.src
        ipv6.dst
        eth.str
        eth.dst
        ip.addr
        ip.src
        ip.dst
        tcp.srcport
        tcp.dstport
        """
        feature_tuples = [
            ("sll.src.eth", "sll.eth"),
            ("ipv6.addr", "ipv6.host"),
            ("eth.src", "eth.src.addr"),
            ("eth.dst", "eth.dst.addr"),
            ("ip.addr", "ip.host"),
        ]
        features_from_list = [
            ("ipv6.src", "ipv6.dst", "ipv6.addr"),
            ("tcp.srcport", "tcp.dstport", "tcp.port"),
            ("ip.src", "ip.dst", "ip.addr"),
        ]

        def ensure_features_exist_func(data_point: dict) -> dict:
            for features in feature_tuples:
                value = next(
                    (
                        data_point[feature]
                        for feature in features
                        if feature in data_point
                    ),
                    None,
                )
                if not value:
                    continue
                new_features = {feature: value for feature in features}
                data_point.update(new_features)

            if "eth.src" in data_point and "eth.dst" in data_point:
                data_point["eth.addr"] = [data_point["eth.src"], data_point["eth.dst"]]

            for src, dst, parent in features_from_list:
                if parent in data_point:
                    data_point[src] = data_point[parent][0]
                    data_point[dst] = data_point[parent][1]
            return data_point

        return self.add_func(lambda d_point: ensure_features_exist_func(d_point))

    @classmethod
    def create_simple_processor(
        cls,
        name: str = "PysharkProcessor",
        log_level: int = None,
        f_features: list[str, ...] = pcap_f_features,
        nn_aggregator: Callable[[str, object], object] = pcap_nn_aggregator,
    ) -> Self:
        """Creates a simple pyshark processor selecting specific features from each
        data point (nan if not existing) and transforms them into numpy vectors,
        ready for to be further processed by detection models.

        :param name: Name of processor for logging purposes.
        :param log_level: Logging level of processor.
        :param f_features: Features to extract from the packets.
        :param nn_aggregator: Aggregator, which should map non-numerical features to
        integers / floats.
        """
        return (
            PysharkProcessor(name=name, log_level=log_level)
            .packet_to_dict()
            .select_dict_features(features=f_features, default_value=np.nan)
            .dict_to_array(nn_aggregator=nn_aggregator)
        )


# noinspection DuplicatedCode
@deprecated("Use DataProcessor.dict_to_array() instead")
def dict_to_numpy_array(
    d_point: dict,
    nn_aggregator: Callable[[str, object], object],
) -> np.ndarray:
    """Transform the pyshark data point directly into a numpy array without further
    processing, aggregating any value that is list into a singular value.

    :param d_point: Data point as dictionary.
    :param nn_aggregator: Aggregator, which maps non-numerical features to integers
    or floats.
    :return: Data point as vector.
    """
    l_point = []
    for key, value in d_point.items():
        if not isinstance(value, int | float):
            value = nn_aggregator(key, value)
        try:
            if np.isnan(value):
                value = 0
        except TypeError as e:
            raise ValueError(f"Invalid k/v pair: {key}, {value}") from e
        l_point.append(value)
    return np.asarray(l_point)


# noinspection DuplicatedCode
@deprecated("Use PysharkProcessor.packet_to_dict() instead")
def packet_to_dict(p: Packet) -> dict:
    """Takes a single pyshark packet and converts it into a dictionary.

    :param p: Packet to convert.
    :return: Dictionary generated from the packet.
    """
    p_dict = {}

    meta_dict = {
        "number": p.number,
        "len": p.length,
        "protocols": [x.layer_name for x in p.layers],
        "time_epoch": p.sniff_timestamp,
        "time": str(p.sniff_time),
    }
    p_dict.update({"meta": meta_dict})

    for layer in p.layers:
        p_dict.update(_add_layer_to_dict(layer))

    return flatten_dict(p_dict)


def _add_layer_to_dict(layer: (XmlLayer, JsonLayer)) -> (dict, list):
    """Creates a dictionary out of a packet captured by pyshark. This is the
    entrypoint for a recursive process.

    :param layer: The base layer of the packet.
    :return: A dictionary containing dictionaries for the sub-layers.
    """
    if isinstance(layer, (XmlLayer, JsonLayer)):
        return _add_xml_layer_to_dict(layer)

    elif isinstance(layer, LayerFieldsContainer):
        return _add_layer_field_container_to_dict(layer)

    elif isinstance(layer, LayerField):
        return {layer.name: layer.raw_value}

    # Backwards Compatibility for JSON-mode
    elif isinstance(layer, list):
        d_list = []
        for sub_layer in layer:
            d_list += [_add_layer_to_dict(sub_layer)]
        return d_list

    else:
        logging.warning("No if case matched")


def _add_xml_layer_to_dict(layer: (XmlLayer, JsonLayer)) -> dict:
    """Creates a dictionary out of a xml layer or json layer and returns it.

    This is part of a recursive function. For the entrypoint see _add_layer_to_dict.

    :param layer: An XML or Json layer from a pyshark packet.
    :return: A dictionary of the given layer.
    """
    dictionary = {}
    for field_name in layer.field_names:
        result_dictionary = _add_layer_to_dict(layer.get_field(field_name))

        # Backwards Compatibility for JSON-mode
        if isinstance(result_dictionary, list):
            dictionary = _add_list_to_dict(layer, field_name, result_dictionary)

        else:
            dictionary.update(result_dictionary)

    layer_dictionary = {layer.layer_name: dictionary}
    return layer_dictionary


def _add_list_to_dict(
    layer: (XmlLayer, JsonLayer), field_name: str, value_list: list
) -> dict:
    """Creates a dictionary out of the given parameters. This function is called by
    _add_xml_layer_to_dict. Only necessary for JSON-mode.

    This is part of a recursive function. For the entrypoint see _add_layer_to_dict.

    :param layer: The XML or JSON layer the value_list is part of.
    :param field_name: The current name of the field in the XML or JSON layer.
    :param value_list: The list in the XML or JSON layer under the name of field_name.
    :return: A dictionary for the field_name and value_list.
    """
    dictionary = {}
    if hasattr(layer.get_field(field_name), "layer_name"):
        dictionary[layer.get_field(field_name).layer_name] = value_list

    else:
        try:
            dictionary[next(iter(value_list[0].keys()))] = [
                res[next(iter(value_list[0].keys()))] for res in value_list
            ]
        except KeyError:
            pass
    return dictionary


def _add_layer_field_container_to_dict(
    layer_field_container: LayerFieldsContainer,
) -> dict:
    """Creates a dictionary out of a layerFieldContainer from a pyshark packet. A
    file in JSON-mode always has a length of one, while XML can contain a list of
    fields.

    This is part of a recursive function. For the entrypoint see _add_layer_to_dict.

    :param layer_field_container: The LayerFieldContainer encountered in the pyshark
    packet.
    :return: A dictionary for the LayerFieldContainer.
    """
    if len(layer_field_container.fields) == 1:
        return _add_layer_to_dict(layer_field_container.fields[0])

    d_list = []
    for field in layer_field_container.fields:
        d_list.append(_add_layer_to_dict(field))

    dictionary = defaultdict(list)
    for d in d_list:
        for key, value in d.items():
            dictionary[key].append(value)

    return dictionary


# noinspection DuplicatedCode
@deprecated("Use PysharkProcessor.create_simple_processor() instead")
def create_pyshark_processor(
    name: str = "PysharkProcessor",
    log_level: int = None,
    f_features: list[str, ...] = pcap_f_features,
    nn_aggregator: Callable[[str, object], object] = pcap_nn_aggregator,
):
    """Creates a DataProcessor using functions specifically for pyshark packets,
    selecting specific features from each data pont (nan if not existing) and
    transforms them into numpy vectors, ready for to be further processed by
    detection models.

    :param name: The name for logging purposes
    :param log_level: Logging level of processor.
    :param f_features: The features to extract from the packets
    :param nn_aggregator: The aggregator, which should map features to integers
    """
    return (
        PysharkProcessor(name=name, log_level=log_level)
        .packet_to_dict()
        .select_dict_features(f_features, default_value=np.nan)
        .dict_to_array(nn_aggregator)
    )


@deprecated("Use DataProcessor.dict_to_json() instead")
def dict_to_json(dictionary: dict) -> str:
    """Takes a dictionary and returns a json object in form of a string.

    :param dictionary: The dictionary to convert to json string.
    :return: A JSON string from the dictionary.
    """
    return json.dumps(dictionary, indent=2)
