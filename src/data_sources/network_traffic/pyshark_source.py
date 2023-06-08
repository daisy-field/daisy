"""
    Implementations of the data source helper interface that allows the processing and provisioning of pyshark packets,
    either via file inputs, live capture, or a remote source that generates packets in either fashion.

    Author: Jonathan Ackerschewski, Fabian Hofmann
    Modified: 08.06.23
"""

import json
import logging
from collections import defaultdict
from collections.abc import MutableMapping
from typing import Iterator

import numpy as np
import pyshark
from pyshark.capture.live_capture import LiveCapture
from pyshark.packet.fields import LayerField
from pyshark.packet.fields import LayerFieldsContainer
from pyshark.packet.layers.json_layer import JsonLayer
from pyshark.packet.layers.xml_layer import XmlLayer
from pyshark.packet.packet import Packet

from src.data_sources.data_source import DataProcessor, SourceHandler

# TODO FILE SOURCE HANDLER (@JONATHAN)
# TODO LIVE CAPTURE FILTER
# TODO TESTING, ADD ARGS
# TODO FIXME COVERAGE CHECKS IN PARSING FUNCTIONS


default_f = (
    'meta.len',
    'meta.time',
    'meta.protocols',
    'ip.addr',
    'sll.halen',
    'sll.pkttype',
    'sll.eth',
    'sll.hatype',
    'sll.unused',
    'ipv6.tclass',
    'ipv6.flow',
    'ipv6.nxt',
    'ipv6.src_host',
    'ipv6.host',
    'ipv6.hlim',
    'sll.ltype',
    'cohda.Type',
    'cohda.Ret',
    'cohda.llc.MKxIFMsg.Ret',
    'ipv6.addr',
    'ipv6.dst',
    'ipv6.plen',
    'tcp.stream',
    'tcp.payload',
    'tcp.urgent_pointer',
    'tcp.port',
    'tcp.options.nop',
    'tcp.options.timestamp',
    'tcp.flags',
    'tcp.window_size_scalefactor',
    'tcp.dstport',
    'tcp.len',
    'tcp.checksum',
    'tcp.window_size',
    'tcp.srcport',
    'tcp.checksum.status',
    'tcp.nxtseq',
    'tcp.status',
    'tcp.analysis.bytes_in_flight',
    'tcp.analysis.push_bytes_sent',
    'tcp.ack',
    'tcp.hdr_len',
    'tcp.seq',
    'tcp.window_size_value',
    'data.data',
    'data.len',
    'tcp.analysis.acks_frame',
    'tcp.analysis.ack_rtt',
    'eth.src.addr',
    'eth.src.eth.src_resolved',
    'eth.src.ig',
    'eth.src.src_resolved',
    'eth.src.addr_resolved',
    'ip.proto',
    'ip.dst_host',
    'ip.flags',
    'ip.len',
    'ip.checksum',
    'ip.checksum.status',
    'ip.version',
    'ip.host',
    'ip.status',
    'ip.id',
    'ip.hdr_len',
    'ip.ttl'
)


class PysharkProcessor(DataProcessor):
    """A simple data processor implementation supporting the processing of pyshark packets.
    """
    f_features: tuple[str, ...]

    def __init__(self, f_features: tuple[str, ...] = default_f):
        """Creates a new pyshark processor.

        :param f_features: Selection of features that every data point will have after processing.
        """
        self.f_features = f_features

    def map(self, o_point: (XmlLayer, JsonLayer)) -> dict:
        """Wrapper around the pyshark packet deserialization functions.

        :param o_point: Data point as pyshark packet.
        :return: Data point as a flattened dictionary.
        """
        return packet_to_dict(o_point)

    def filter(self, d_point: dict) -> dict:
        """Filters the pyshark packet according to a pre-defined filter which is applied to every dictionary in order of
        the selected features in the filter. Features that do not exist are set to None.

        :param d_point: Data point as dictionary.
        :return: Data point as dictionary, ordered.
        """
        return {f_feature: d_point.pop(f_feature, None) for f_feature in self.f_features}

    def reduce(self, d_point: dict) -> np.ndarray:
        """Transform the pyshark data point directly into a numpy array without further processing.

        :param d_point: Data point as dictionary.
        :return: Data point as vector.
        """
        return np.asarray(list(d_point.values()))


class LivePysharkHandler(SourceHandler):
    """The wrapper implementation to support and handle pyshark live captures as data sources. Considered infinite in
    nature, as it allows the generation of pyshark packets, until the capture is stopped.
    """
    _capture: LiveCapture
    _generator: Iterator[Packet]

    def __init__(self, interfaces: list = 'any'):
        """Creates a new basic pyshark live capture handler on the given interfaces.

        :param interfaces: Network interfaces to capture. If not given, runs on all interfaces.
        """
        self._capture = pyshark.LiveCapture(interface=interfaces)

    def open(self):
        """Starts the pyshark live caption, initializing the wrapped generator.
        """
        self._generator = self._capture.sniff_continuously()

    def close(self):
        """Stops the live caption, essentially disabling the generator. Note that the generator might block if one
        tries to retrieve an object from it after that point.
        """
        self._capture.close()

    def __iter__(self) -> Iterator[Packet]:
        """Returns the wrapped generator. Note this does not catch problems after a close() on the handler is called ---
        one must not retrieve objects after as it will result in a deadlock!

        :return: Pyshark generator object for data points as pyshark packets.
        """
        return self._generator


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
        "time": str(p.sniff_time)
    }
    p_dict.update({"meta": meta_dict})

    for layer in p.layers:
        p_dict.update(_add_layer_to_dict(layer))

    return flatten_dict(p_dict)


def _add_layer_to_dict(layer: (XmlLayer, JsonLayer)) -> (dict, list):
    """Creates a dictionary out of a packet captured by pyshark. This is the entrypoint for a recursive process.

    :param layer: The base layer of the packet.
    :return: A dictionary containing dictionaries for the sub-layers.
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
    """Creates a dictionary out of a xml layer or json layer and returns it.
    This is part of a recursive function. For the entrypoint see _add_layer_to_dict.

    :param layer: An XML or Json layer from a pyshark packet.
    :return: A dictionary of the given layer.
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
    """Creates a dictionary out of the given parameters. This function is called by _add_xml_layer_to_dict.
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
        dictionary[next(iter(value_list[0].keys()))] = \
            [res[next(iter(value_list[0].keys()))] for res in value_list]
    return dictionary


def _add_layer_field_container_to_dict(layer_field_container: LayerFieldsContainer) -> dict:
    """Creates a dictionary out of a layerFieldContainer from a pyshark packet.
    This is part of a recursive function. For the entrypoint see _add_layer_to_dict.

    :param layer_field_container: The LayerFieldContainer encountered in the pyshark packet.
    :return: A dictionary for the LayerFieldContainer.
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
    """Creates a flat dictionary (a dictionary without sub-dictionaries) from the given dictionary. The keys of
    sub-dictionaries are merged into the parent dictionary by combining the keys and adding a seperator:
    {a: {b: c, d: e}, f: g} becomes {a.b: c, a.d: e, f: g} assuming the seperator as '.'

    :param dictionary: The dictionary to flatten.
    :param seperator: The seperator to use.
    :param par_key: The key of the parent dictionary.
    :return: A flat dictionary with keys merged and seperated using the seperator.
    """
    items = {}
    for key, val in dictionary.items():
        cur_key = par_key + seperator + key if par_key else key
        if isinstance(val, MutableMapping):
            items.update(flatten_dict(val, par_key=cur_key, seperator=seperator))
        else:
            items.update({cur_key: val})
    return items


def dict_to_json(dictionary: dict) -> str:  # FIXME CAN THIS BE REMOVED?
    """Takes a dictionary and returns a json object in form of a string.

    :param dictionary: The dictionary to convert to json string.
    :return: A JSON string from the dictionary.
    """
    return json.dumps(dictionary, indent=2)

# if __name__ == '__main__':
#     logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
#                         level=logging.DEBUG)
#
#     count = 0
#     with RemoteTrafficSourceHandler(multithreading=True) as rts:
#         for packet in rts:
#             count += 1
#             if count > 16:
#                 break
#             logging.info(f"Received Pyshark Packet: {packet}")
