import json
import logging
import ipaddress
from typing import Callable
from ipaddress import AddressValueError
import numpy as np
from collections import defaultdict
from collections.abc import MutableMapping

from pyshark.packet.fields import LayerField, LayerFieldsContainer
from pyshark.packet.layers.json_layer import JsonLayer
from pyshark.packet.layers.xml_layer import XmlLayer
from pyshark.packet.packet import Packet

from daisy.data_sources import SimpleDataProcessor

# Exemplary network feature filter, supporting cohda-box (V2x) messages, besides TCP/IP and ETH.
default_f_features = (
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


def default_nn_aggregator(key: str, value: object) -> int:
    """Simple, exemplary value aggregator. Takes a non-numerical key-value pair and attempts to converted it into an
    integer. This example does not take the key into account, but only checks the types of the value to proceed.

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
            return int(ipaddress.IPv6Address(value))
        except AddressValueError:
            pass
        try:
            return int(value, 16)
        except ValueError:
            pass
        return hash(value)

    raise ValueError(f"Unable to aggregate non-numerical item: {key, value}")
# TODO comments

def create_pyshark_processor(name: str = "", f_features: tuple[str, ...] = default_f_features,  # TODO comment
                 nn_aggregator: Callable[[str, object], object] = default_nn_aggregator):
    return SimpleDataProcessor(pyshark_map_fn(), pyshark_filter_fn(f_features), pyshark_reduce_fn(nn_aggregator), name)


# TODO is there a naming convention for functions, that return functions?
def pyshark_map_fn() -> Callable[[object], dict]:  # TODO comment
    """Wrapper around the pyshark packet deserialization functions.

            :param o_point: Data point as pyshark packet.
            :return: Data point as a flattened dictionary.
            """
    return lambda o_point: packet_to_dict(o_point)


def pyshark_filter_fn(f_features: tuple[str, ...] = default_f_features) -> Callable[[dict], dict]:  # TODO comment
    return lambda d_point: _pyshark_filter_fn(d_point, f_features)


def _pyshark_filter_fn(d_point: dict, f_features: tuple[str, ...]) -> dict: # TODO comment
    """Filters the pyshark packet according to a pre-defined filter which is applied to every dictionary in order of
    the selected features in the filter. Features that do not exist are set to None.

    :param d_point: Data point as dictionary.
    :return: Data point as dictionary, ordered.
    """
    return {f_feature: d_point.pop(f_feature, np.nan) for f_feature in f_features}


def pyshark_reduce_fn(nn_aggregator: Callable[[str, object], object] = default_nn_aggregator) -> Callable[[dict], np.ndarray]:  # TODO usage: SimpleMethodDataProcessor.__init__(_,_, pyshark_reduce_fn(some_nn_aggregator))
    """Transform the pyshark data point directly into a numpy array without further processing, aggregating any  # TODO comment
    value that is list into a singular value.

    :param d_point: Data point as dictionary.
    :return: Data point as vector.
    """
    return lambda d_point: _pyshark_reduce_fn(d_point, nn_aggregator)


def _pyshark_reduce_fn(d_point: dict, nn_aggregator: Callable[[str, object], object]) -> np.ndarray:  # TODO comments
    """Transform the pyshark data point directly into a numpy array without further processing, aggregating any
        value that is list into a singular value.

        :param d_point: Data point as dictionary.
        :return: Data point as vector.
        """
    l_point = []
    for key, value in d_point.items():
        if not isinstance(value, int | float):
            value = nn_aggregator(key, value)
        if np.isnan(value):
            value = 0
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


def _add_list_to_dict(layer: (XmlLayer, JsonLayer), field_name: str, value_list: list) -> dict:
    """Creates a dictionary out of the given parameters. This function is called by _add_xml_layer_to_dict. Only
    necessary for JSON-mode.
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
    """Creates a dictionary out of a layerFieldContainer from a pyshark packet. A file in JSON-mode always has a length
    of one, while XML can contain a list of fields.
    This is part of a recursive function. For the entrypoint see _add_layer_to_dict.

    :param layer_field_container: The LayerFieldContainer encountered in the pyshark packet.
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


def flatten_dict(dictionary: (dict, list), seperator: str = ".", par_key: str = "") -> dict:
    """Creates a flat dictionary (a dictionary without sub-dictionaries) from the given dictionary. The keys of
    sub-dictionaries are merged into the parent dictionary by combining the keys and adding a seperator:
    {a: {b: c, d: e}, f: g} becomes {a.b: c, a.d: e, f: g} assuming the seperator as '.'. However, redundant parent keys
    are greedily eliminated from the dictionary.

    :param dictionary: The dictionary to flatten.
    :param seperator: The seperator to use.
    :param par_key: The key of the parent dictionary.
    :return: A flat dictionary with keys merged and seperated using the seperator.
    :raises ValueError: If there are key-collisions by greedily flattening the dictionary.
    """
    items = {}
    for key, val in dictionary.items():
        cur_key = par_key + seperator + key if par_key != "" and not key.startswith(par_key + seperator) else key
        if isinstance(val, MutableMapping):
            sub_items = flatten_dict(val, par_key=cur_key, seperator=seperator)
            for subkey in sub_items.keys():
                if subkey in items:
                    raise ValueError(f"Key collision in dictionary "
                                     f"({subkey, sub_items[subkey]} vs {subkey, items[subkey]})!")
            items.update(sub_items)
        else:
            if cur_key in items:
                raise ValueError(f"Key collision in dictionary ({cur_key, val} vs {cur_key, items[cur_key]})!")
            items.update({cur_key: val})
    return items


def dict_to_json(dictionary: dict) -> str:
    """Takes a dictionary and returns a json object in form of a string.

    :param dictionary: The dictionary to convert to json string.
    :return: A JSON string from the dictionary.
    """
    return json.dumps(dictionary, indent=2)
