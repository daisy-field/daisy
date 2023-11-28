"""
    Implementations of the data source helper interface that allows the processing and provisioning of pyshark packets,
    either via file inputs, live capture, or a remote source that generates packets in either fashion.

    Author: Jonathan Ackerschewski, Fabian Hofmann
    Modified: 08.06.23

    # TODO Future Work: Encoding/mapping of string values into numerical features
    # TODO Future Work: ALT.: Flattening of Lists instead of encoding them into singular numerical features
"""

import json
import logging
import os
from collections import defaultdict
from collections.abc import MutableMapping
from typing import Callable, Iterator, Optional

import numpy as np
import pyshark
from pyshark.capture.capture import TSharkCrashException
from pyshark.capture.file_capture import FileCapture
from pyshark.capture.live_capture import LiveCapture
from pyshark.packet.fields import LayerField, LayerFieldsContainer
from pyshark.packet.layers.json_layer import JsonLayer
from pyshark.packet.layers.xml_layer import XmlLayer
from pyshark.packet.packet import Packet

from src.data_sources.data_source import DataProcessor, SourceHandler

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


def default_l_aggregator(key: str, value_l: list) -> int:
    value_l.sort()
    return hash(str(value_l))


class PysharkProcessor(DataProcessor):
    """A simple data processor implementation supporting the processing of pyshark packets.
    """
    f_features: tuple[str, ...]
    l_aggregator: Callable[[str, list], object]

    def __init__(self, name: str = "", f_features: tuple[str, ...] = default_f,
                 l_aggregator: Callable[[str, list], object] = default_l_aggregator):
        """Creates a new pyshark processor.

        :param name: Name of processor for logging purposes.
        :param f_features: Selection of features that every data point will have after processing.
        :param l_aggregator: List aggregator that is able to aggregator dictionary values that are lists into singleton
        values, depending on the key they are sorted under.
        """
        super().__init__(name)

        self.f_features = f_features
        self.l_aggregator = l_aggregator

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
        return {f_feature: d_point.pop(f_feature, np.nan) for f_feature in self.f_features}

    def reduce(self, d_point: dict) -> np.ndarray:
        """Transform the pyshark data point directly into a numpy array without further processing, aggregating any
        value that is list into a singular value.

        :param d_point: Data point as dictionary.
        :return: Data point as vector.
        """
        l_point = []
        for key, value in d_point.items():
            if isinstance(value, list): # FIXME
                value = self.l_aggregator(key, value)
            l_point.append(value)
        return np.asarray(l_point)


class LivePysharkHandler(SourceHandler):
    """The wrapper implementation to support and handle pyshark live captures as data sources. Considered infinite in
    nature, as it allows the generation of pyshark packets, until the capture is stopped.
    """
    _capture: LiveCapture
    _generator: Iterator[Packet]

    def __init__(self, name: str = "", interfaces: list = 'any', bpf_filter: str = ""):
        """Creates a new basic pyshark live capture handler on the given interfaces.

        :param name: Name of handler for logging purposes.
        :param interfaces: Network interfaces to capture. If not given, runs on all interfaces.
        :param bpf_filter: Pcap conform filter to filter or ignore certain traffic.
        """
        super().__init__(name)

        self._logger.info("Initializing live pyshark handler...")
        self._capture = pyshark.LiveCapture(interface=interfaces, bpf_filter=bpf_filter)
        self._logger.info("Live pyshark handler initialized.")

    def open(self):
        """Starts the pyshark live caption, initializing the wrapped generator.
        """
        self._logger.info("Beginning live pyshark capture...")
        self._generator = self._capture.sniff_continuously()

    def close(self):
        """Stops the live caption, essentially disabling the generator. Note that the generator might block if one
        tries to retrieve an object from it after that point.
        """
        self._capture.close()
        self._logger.info("Live pyshark capture stopped.")

    def __iter__(self) -> Iterator[Packet]:
        """Returns the wrapped generator. Note this does not catch problems after a close() on the handler is called ---
        one must not retrieve objects after as it will result in a deadlock!

        :return: Pyshark generator object for data points as pyshark packets.
        """
        return self._generator


class PcapHandler(SourceHandler):
    """The wrapper implementation to support and handle any number of pcap files as data sources. Finite: finishes after
    all files have been processed. Warning: Not entirely compliant with the source handler abstract class: Neither
    fully thread safe, nor does its __iter__() method shut down after close() has been called. Due to its finite nature
    acceptable however, as this handler is nearly always only closed ones all data points have been retrieved.
    """
    _pcap_files: list[str]

    _cur_file_counter: int
    _cur_file_handle: Optional[FileCapture]
    _try_counter: int

    def __init__(self, *file_names: str, try_counter: int = 3, name: str = ""):
        """Creates a new pcap file handler.

        :param file_names: List of paths of single files or directories containing .pcap files. Each string should be a
        name of a file or directory. In case a directory is passed, all files ending in .pcap are used. In case a single
        file is passed, it is used regardless of file ending.
        :param try_counter: Number of attempts to open a specific pcap file until throwing an exception.
        :param name: Name of handler for logging purposes.
        """
        super().__init__(name)

        self._logger.info("Initializing pcap file handler...")
        self._pcap_files = []
        for path in file_names:
            if os.path.isdir(path):
                # Variables in following line are: file_tuple[0] = <sub>-directories; file_tuple[2] = files in directory
                dirs = [(file_tuple[0], file_tuple[2]) for file_tuple in os.walk(path)]
                files = [os.path.join(file_tuple[0], file_name) for file_tuple in dirs for file_name in file_tuple[1]
                         if file_name.endswith(".pcap")]
                if files is None:
                    raise ValueError(f"Directory '{path}' does not contain any .pcap files!")
                self._pcap_files += files
            elif os.path.isfile(path) and path.endswith(".pcap"):
                self._pcap_files.append(path)
        if not self._pcap_files:
            raise ValueError(f"No .pcap files in '{file_names}' could be found.")

        self._cur_file_counter = 0
        self._cur_file_handle = None
        self._try_counter = try_counter
        self._logger.info("Pcap file handler initialized.")

    def open(self):
        """Opens and resets the pcap file handler to the very beginning of the file list.
        """
        self._logger.info("Opening pcap file source...")
        self._cur_file_counter = 0
        self._cur_file_handle = None
        self._logger.info("Pcap file source opened.")

    def close(self):
        """Closes any file of the pcap file handler.
        """
        self._logger.info("Closing pcap file source...")
        if self._cur_file_handle is not None:
            self._cur_file_handle.close()
        self._logger.info("Pcap file source closed.")

    def _open(self):
        """Opens the next file of the pcap file list, trying to open it several times until succeeding (known bug from
        the pyshark library).
        """
        self._logger.debug("Opening next pcap file...")
        try_counter = 0
        while try_counter < self._try_counter:
            try:
                self._cur_file_handle = pyshark.FileCapture(self._pcap_files[self._cur_file_counter])
                break
            except TSharkCrashException:
                try_counter += 1
                continue
        if try_counter == self._try_counter:
            raise RuntimeError(f"Could not open File '{self._pcap_files[self._cur_file_counter]}'")
        self._cur_file_counter += 1
        self._logger.info("Next pcap file opened.")

    def __iter__(self) -> Iterator[Packet]:
        """Returns a generator that yields pyshark packets from each file after another, opening and closing them when
        being actively read.

        :return: Generator object for data points as pyshark packets.
        """
        for _ in self._pcap_files:
            self._open()
            for packet in self._cur_file_handle:
                yield packet
            self._cur_file_handle.close()


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
