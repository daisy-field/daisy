"""
    TODO

    Author: Jonathan Ackerschewski, Fabian Hofmann
    Modified: 13.04.22
"""

import json
import logging
import pickle
from collections import defaultdict
from collections.abc import MutableMapping

import numpy as np
from pyshark.packet.fields import LayerField
from pyshark.packet.fields import LayerFieldsContainer
from pyshark.packet.layers.xml_layer import XmlLayer
from pyshark.packet.layers.json_layer import JsonLayer
from pyshark.packet.packet import Packet

import communication.message_stream as stream
from data_sources.data_source import DataSource, RemoteDataSource

# TODO add args to main for deployment


# new_dict = {
#     'meta.len': 0,
#     'meta.time': 0,
#     'meta.protocols': 0,
#     'ip.addr': 0,
#     'sll.halen': 0,
#     'sll.pkttype': 0,
#     'sll.eth': 0,
#     'sll.hatype': 0,
#     'sll.unused': 0,
#     'ipv6.addr': 0,
#     'ipv6.plen': 0,
#     'ipv6.tclass': 0,
#     'ipv6.flow': 0,
#     'ipv6.dst': 0,
#     'ipv6.nxt': 0,
#     'ipv6.src_host': 0,
#     'ipv6.host': 0,
#     'ipv6.hlim': 0,
#     'tcp.window_size_scalefactor': 0,
#     'tcp.checksum.status': 0,
#     'tcp.analysis.bytes_in_flight': 0,
#     'tcp.analysis.push_bytes_sent': 0,
#     'tcp.payload': 0,
#     'tcp.port': 0,
#     'tcp.len': 0,
#     'tcp.hdr_len': 0,
#     'tcp.window_size': 0,
#     'tcp.checksum': 0,
#     'tcp.ack': 0,
#     'tcp.srcport': 0,
#     'tcp.stream': 0,
#     'tcp.dstport': 0,
#     'tcp.seq': 0,
#     'tcp.window_size_value': 0,
#     'tcp.status': 0,
#     'tcp.urgent_pointer': 0,
#     'tcp.nxtseq': 0,
#     'data.data': 0,
#     'data.len': 0,
#     'tcp.analysis.acks_frame': 0,
#     'tcp.analysis.ack_rtt': 0,
#     'sll.ltype': 0,
#     'cohda.Type': 0,
#     'cohda.Ret': 0,
#     'cohda.llc.MKxIFMsg.Ret': 0,
#     'ipv6.addr': 0,
#     'ipv6.dst': 0,
#     'ipv6.plen': 0,
#     'tcp.stream': 0,
#     'tcp.payload': 0,
#     'tcp.urgent_pointer': 0,
#     'tcp.port': 0,
#     'tcp.options.nop': 0,
#     'tcp.options.timestamp': 0,
#     'tcp.flags': 0,
#     'tcp.window_size_scalefactor': 0,
#     'tcp.dstport': 0,
#     'tcp.len': 0,
#     'tcp.checksum': 0,
#     'tcp.window_size': 0,
#     'tcp.srcport': 0,
#     'tcp.checksum.status': 0,
#     'tcp.nxtseq': 0,
#     'tcp.status': 0,
#     'tcp.analysis.bytes_in_flight': 0,
#     'tcp.analysis.push_bytes_sent': 0,
#     'tcp.ack': 0,
#     'tcp.hdr_len': 0,
#     'tcp.seq': 0,
#     'tcp.window_size_value': 0,
#     'data.data': 0,
#     'data.len': 0,
#     'tcp.analysis.acks_frame': 0,
#     'tcp.analysis.ack_rtt': 0,
#     'eth.src.addr': 0,
#     'eth.src.eth.src_resolved': 0,
#     'eth.src.ig': 0,
#     'eth.src.src_resolved': 0,
#     'eth.src.addr_resolved': 0,
#     'ip.proto': 0,
#     'ip.dst_host': 0,
#     'ip.flags': 0,
#     'ip.len': 0,
#     'ip.checksum': 0,
#     'ip.checksum.status': 0,
#     'ip.version': 0,
#     'ip.host': 0,
#     'ip.status': 0,
#     'ip.id': 0,
#     'ip.hdr_len': 0,
#     'ip.ttl': 0
# }


class TrafficSource(DataSource):
    """
    TODO
    """

    def reduce(self, d_point: dict) -> np.ndarray:
        """
        TODO
        :param d_point:
        :return:
        """
        return d_point
        #return np.asarray(d_point.values())

    def filter(self, d_point: dict) -> dict:
        """
        TODO
        :param d_point:
        :return:
        """
        return d_point

    def map(self, o_point: (XmlLayer, JsonLayer)) -> dict:
        """
        TODO
        :param o_point:
        :return:
        """
        return packet_to_dict(o_point)


class RemoteTrafficSource(RemoteDataSource):
    """
    TODO
    """

    def reduce(self, d_point: dict) -> np.ndarray:
        """
        TODO
        :param d_point:
        :return:
        """
        return d_point
        #return np.asarray(d_point.values())

    def filter(self, d_point: dict) -> dict:
        """
        TODO
        :param d_point:
        :return:
        """
        return d_point

    def map(self, o_point: (XmlLayer, JsonLayer)) -> dict:
        """
        TODO
        :param o_point:
        :return:
        """
        return packet_to_dict(o_point)


def _add_layer_to_dict(layer: (XmlLayer, JsonLayer)) -> (dict, list):
    """
    Creates a dictionary out of a packet captured by PyShark. This is the entrypoint for a recursive process.

    :param layer: The base layer of the packet
    :return: A dictionary containing dictionaries for the sub-layers
    """
    if isinstance(layer, (XmlLayer, JsonLayer)):  # FIXME TEST DIFFERENT MODI
        return _add_xml_layer_to_dict(layer)

    elif isinstance(layer, LayerFieldsContainer):
        return _add_layer_field_container_to_dict(layer)

    elif isinstance(layer, LayerField):
        return {layer.name: layer.show}

    elif isinstance(layer, list):  # FIXME no coverage in XML mode -> check with other PCAPs
        d_list = []
        for sub_layer in layer:
            d_list += [_add_layer_to_dict(sub_layer)]
        return d_list

    else:
        logging.warning("No if case matched")


def _add_xml_layer_to_dict(layer: (XmlLayer, JsonLayer)) -> dict:
    """
    Creates a dictionary out of a xml layer or json layer and returns it.
    This is part of a recursive function. For the entrypoint see _add_layer_to_dict.

    :param layer: An XML or Json layer from a PyShark packet.
    :return: A dictionary of the given layer
    """
    dictionary = {}
    for field_name in layer.field_names:
        result_dictionary = _add_layer_to_dict(layer.get_field(field_name))

        if isinstance(result_dictionary, list):  # FIXME no coverage in XML mode -> check with other PCAPs
            dictionary = _add_list_to_dict(layer, field_name, result_dictionary)

        else:
            dictionary.update(result_dictionary)

    layer_dictionary = {layer.layer_name: dictionary}
    return layer_dictionary


def _add_list_to_dict(layer: (XmlLayer, JsonLayer), field_name: str, value_list: list) -> dict:
    """
    Creates a dictionary out of the given parameters. This function is called by _add_xml_layer_to_dict.
    This is part of a recursive function. For the entrypoint see _add_layer_to_dict.

    :param layer: The XML or JSON layer the value_list is part of
    :param field_name: The current name of the field in the XML or JSON layer
    :param value_list: The list in the XML or JSON layer under the name of field_name
    :return: A dictionary for the field_name and value_list
    """
    dictionary = {}
    if hasattr(layer.get_field(field_name), "layer_name"):
        dictionary[layer.get_field(field_name).layer_name] = value_list

    else:
        dictionary[next(iter(value_list[0].keys()))] = \
            [res[next(iter(value_list[0].keys()))] for res in value_list]
    return dictionary


def _add_layer_field_container_to_dict(layer_field_container: LayerFieldsContainer) -> dict:
    """
    Creates a dictionary out of a layerFieldContainer from a PyShark packet.
    This is part of a recursive function. For the entrypoint see _add_layer_to_dict.

    :param layer_field_container: The LayerFieldContainer encountered in the PyShark packet
    :return: A dictionary for the LayerFieldContainer
    """
    if len(layer_field_container.fields) == 1:
        return _add_layer_to_dict(layer_field_container.fields[0])

    d_list = []  # FIXME only necessary for XML mode (json mode only has == 1)
    for field in layer_field_container.fields:
        d_list.append(_add_layer_to_dict(field))

    dictionary = defaultdict(list)
    for d in d_list:
        for key, value in d.items():
            dictionary[key].append(value)

    return dictionary


def flatten_dict(dictionary: (dict, list), seperator: str = ".", par_key: str = "") -> dict:
    """
    Creates a flat dictionary (a dictionary without sub-dictionaries) from the given dictionary. The keys of
    sub-dictionaries are merged into the parent dictionary by combining the keys and adding a seperator:
    {a: {b: c, d: e}, f: g} becomes {a.b: c, a.d: e, f: g} assuming the seperator as '.'

    :param dictionary: The dictionary to flatten
    :param seperator: The seperator to use
    :param par_key: The key of the parent dictionary
    :return: A flat dictionary with keys merged and seperated using the seperator
    """
    items = {}
    for key, val in dictionary.items():
        cur_key = par_key + seperator + key if par_key else key
        if isinstance(val, MutableMapping):
            items.update(flatten_dict(val, par_key=cur_key, seperator=seperator))
        else:
            items.update({cur_key: val})
    return items


def dict_to_json(dictionary: dict) -> str:
    """
    Takes a dictionary and returns a json object in form of a string.

    :param dictionary: The dictionary to convert to json string
    :return: A JSON string from the dictionary
    """
    return json.dumps(dictionary, indent=2)


def packet_to_dict(p: Packet) -> dict:
    """
    Takes a single PyShark packet and converts it into a dictionary.

    :param p: The packet to convert
    :return: The dictionary generated from the packet
    """
    p_dict = {}

    meta_dict = {
        "number": p.number,
        "len": p.length,
        "protocols": [x.layer_name for x in p.layers],
        "time_epoch": p.sniff_timestamp,
        "time": str(p.sniff_time)
    }
    p_dict.update({"meta": meta_dict})

    for layer in p.layers:
        p_dict.update(_add_layer_to_dict(layer))

    return flatten_dict(p_dict)


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.DEBUG)

    count = 0
    with RemoteTrafficSource(multithreading=True) as rts:
        for packet in rts:
            count += 1
            if count > 16:
                break
            logging.info(f"Received Pyshark Packet: {packet}")
