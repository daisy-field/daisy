# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Implementation of processing steps used for pyshark packets. Also includes a
pre-packaged extension of the data processor base class for ease of use.

Author: Jonathan Ackerschewski, Fabian Hofmann
Modified: 04.11.2024
"""
# TODO: Future Work:
#   - Encoding/mapping of string/non-numerical values into numerical features
#   - Flattening of Lists instead of encoding them into singular numerical features
#   - NaN values also need to converted to something useful
#     (that does not break the prediction/training)

import ipaddress
import json
import logging
import sys
from collections import defaultdict
from ipaddress import AddressValueError
from typing import Callable

import numpy as np
from pyshark.packet.fields import LayerField, LayerFieldsContainer
from pyshark.packet.layers.json_layer import JsonLayer
from pyshark.packet.layers.xml_layer import XmlLayer
from pyshark.packet.packet import Packet
from warnings import deprecated

from .. import select_feature
from ..data_processor import DataProcessor, flatten_dict


# Exemplary network feature filter, supporting cohda-box (V2x) messages, besides
# TCP/IP and ETH.
default_f_features = (
    "meta.len",
    "meta.time",
    "meta.protocols",
    "ip.addr",
    "sll.halen",
    "sll.pkttype",
    "sll.eth",
    "sll.hatype",
    "sll.unused",
    "ipv6.tclass",
    "ipv6.flow",
    "ipv6.nxt",
    "ipv6.src_host",
    "ipv6.host",
    "ipv6.hlim",
    "sll.ltype",
    "cohda.Type",
    "cohda.Ret",
    "cohda.llc.MKxIFMsg.Ret",
    "ipv6.addr",
    "ipv6.dst",
    "ipv6.plen",
    "tcp.stream",
    "tcp.payload",
    "tcp.urgent_pointer",
    "tcp.port",
    "tcp.options.nop",
    "tcp.options.timestamp",
    "tcp.flags",
    "tcp.window_size_scalefactor",
    "tcp.dstport",
    "tcp.len",
    "tcp.checksum",
    "tcp.window_size",
    "tcp.srcport",
    "tcp.checksum.status",
    "tcp.nxtseq",
    "tcp.status",
    "tcp.analysis.bytes_in_flight",
    "tcp.analysis.push_bytes_sent",
    "tcp.ack",
    "tcp.hdr_len",
    "tcp.seq",
    "tcp.window_size_value",
    "data.data",
    "data.len",
    "tcp.analysis.acks_frame",
    "tcp.analysis.ack_rtt",
    "eth.src.addr",
    "eth.src.eth.src_resolved",
    "eth.src.ig",
    "eth.src.src_resolved",
    "eth.src.addr_resolved",
    "ip.proto",
    "ip.dst_host",
    "ip.flags",
    "ip.len",
    "ip.checksum",
    "ip.checksum.status",
    "ip.version",
    "ip.host",
    "ip.status",
    "ip.id",
    "ip.hdr_len",
    "ip.ttl",
)


def default_nn_aggregator(key: str, value: object) -> int | float:
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


def create_pyshark_processor(
    name: str = "",
    f_features: list[str, ...] = default_f_features,
    nn_aggregator: Callable[[str, object], object] = default_nn_aggregator,
):
    """Creates a DataProcessor using functions specifically for pyshark packets,
    selecting specific features from each data pont (nan if not existing) and
    transforms them into numpy vectors, ready for to be further processed by
    detection models.

    :param name: The name for logging purposes
    :param f_features: The features to extract from the packets
    :param nn_aggregator: The aggregator, which should map features to integers
    """
    return (
        DataProcessor(name=name)
        .add_func(lambda o_point: packet_to_dict(o_point))
        .add_func(
            lambda o_point: select_feature(
                d_point=o_point, f_features=f_features, default_value=np.nan
            )
        )
        .add_func(
            lambda o_point: dict_to_numpy_array(
                d_point=o_point, nn_aggregator=nn_aggregator
            )
        )
    )


def dict_to_numpy_array(
    d_point: dict,
    nn_aggregator: Callable[[str, object], object],
) -> np.ndarray:
    """Transform the pyshark data point directly into a numpy array without further
    processing, aggregating any value that is list into a singular value.

    :param d_point: Data point as dictionary.
    :param nn_aggregator: The aggregator, which should map features to integers
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


def packet_to_dict(p: Packet) -> dict:
    """Takes a single pyshark packet and converts it into a dictionary.

    :param p: The packet to convert.
    :return: The dictionary generated from the packet.
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
        return {layer.name: layer.show}

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
        dictionary[next(iter(value_list[0].keys()))] = [
            res[next(iter(value_list[0].keys()))] for res in value_list
        ]
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


@deprecated("Use DataProcessor.to_json() instead")
def dict_to_json(dictionary: dict) -> str:
    """Takes a dictionary and returns a json object in form of a string.

    :param dictionary: The dictionary to convert to json string.
    :return: A JSON string from the dictionary.
    """
    return json.dumps(dictionary, indent=2)
