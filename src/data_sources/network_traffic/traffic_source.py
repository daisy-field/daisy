"""
    TODO

    Author: Jonathan Ackerschewski, Fabian Hofmann
    Modified: 13.04.22
"""

import json
import logging
import pickle
import typing
from collections import defaultdict
from collections.abc import MutableMapping

from pyshark.packet.fields import LayerField
from pyshark.packet.fields import LayerFieldsContainer
from pyshark.packet.layers.xml_layer import XmlLayer
from pyshark.packet.packet import Packet

from abc import ABC

import communication.message_stream as stream

# TODO refactor via abstract classes, 2 versions (remote, local)
# TODO further cleanup, docstrings, typehints
# TODO add args to main for deployment



new_dict = {
			'meta.len': 0,
			'meta.time': 0,
			'meta.protocols': 0,
			'ip.addr': 0,
			'sll.halen': 0,
			'sll.pkttype': 0,
			'sll.eth': 0,
			'sll.hatype': 0,
			'sll.unused': 0,
			'ipv6.addr': 0,
			'ipv6.plen': 0,
			'ipv6.tclass': 0,
			'ipv6.flow': 0,
			'ipv6.dst': 0,
			'ipv6.nxt': 0,
			'ipv6.src_host': 0,
			'ipv6.host': 0,
			'ipv6.hlim': 0,
			'tcp.window_size_scalefactor': 0,
			'tcp.checksum.status': 0,
			'tcp.analysis.bytes_in_flight': 0,
			'tcp.analysis.push_bytes_sent': 0,
			'tcp.payload': 0,
			'tcp.port': 0,
			'tcp.len': 0,
			'tcp.hdr_len': 0,
			'tcp.window_size': 0,
			'tcp.checksum': 0,
			'tcp.ack': 0,
			'tcp.srcport': 0,
			'tcp.stream': 0,
			'tcp.dstport': 0,
			'tcp.seq': 0,
			'tcp.window_size_value': 0,
			'tcp.status': 0,
			'tcp.urgent_pointer': 0,
			'tcp.nxtseq': 0,
			'data.data': 0,
			'data.len': 0,
			'tcp.analysis.acks_frame': 0,
			'tcp.analysis.ack_rtt': 0,
			'sll.ltype': 0,
			'cohda.Type': 0,
			'cohda.Ret': 0,
			'cohda.llc.MKxIFMsg.Ret': 0,
			'ipv6.addr': 0,
			'ipv6.dst': 0,
			'ipv6.plen': 0,
			'tcp.stream': 0,
			'tcp.payload': 0,
			'tcp.urgent_pointer': 0,
			'tcp.port': 0,
			'tcp.options.nop': 0,
			'tcp.options.timestamp': 0,
			'tcp.flags': 0,
			'tcp.window_size_scalefactor': 0,
			'tcp.dstport': 0,
			'tcp.len': 0,
			'tcp.checksum': 0,
			'tcp.window_size': 0,
			'tcp.srcport': 0,
			'tcp.checksum.status': 0,
			'tcp.nxtseq': 0,
			'tcp.status': 0,
			'tcp.analysis.bytes_in_flight': 0,
			'tcp.analysis.push_bytes_sent': 0,
			'tcp.ack': 0,
			'tcp.hdr_len': 0,
			'tcp.seq': 0,
			'tcp.window_size_value': 0,
			'data.data': 0,
			'data.len': 0,
			'tcp.analysis.acks_frame': 0,
			'tcp.analysis.ack_rtt': 0,
			'eth.src.addr': 0,
			'eth.src.eth.src_resolved': 0,
			'eth.src.ig': 0,
			'eth.src.src_resolved': 0,
			'eth.src.addr_resolved': 0,
			'ip.proto': 0,
			'ip.dst_host': 0,
			'ip.flags': 0,
			'ip.len': 0,
			'ip.checksum': 0,
			'ip.checksum.status': 0,
			'ip.version': 0,
			'ip.host': 0,
			'ip.status': 0,
			'ip.id': 0,
			'ip.hdr_len': 0,
			'ip.ttl'
			}


def add_layer_to_dict(layer):
    if isinstance(layer, XmlLayer):
        dictionary = {}
        for field in layer.field_names:
            result = add_layer_to_dict(layer.get_field(field))
            if isinstance(result, list):  # FIXME no coverage in XML mode -> check with other PCAPs
                if hasattr(layer.get_field(field), "layer_name"):
                    dictionary.update({layer.get_field(field).layer_name: result})
                else:
                    # dictionary.update({f"RandomFieldName{randint(100000, 999999)}": result})
                    # Assumptions:
                    # - result has no layer_name, because it is a list
                    # - the list is never empty
                    # - all elements inside the list are dictionaries
                    # - all dictionaries inside the list have one element
                    # - all dictionaries inside the list share the same key
                    # If this fails, try enabling the RandomFieldName above and look in the generated json
                    # what the result looks like and which assumption doesn't hold
                    dictionary.update(
                        {next(iter(result[0].keys())): [res[next(iter(result[0].keys()))] for res in result]})
            else:
                dictionary.update(result)
        layer_list = {layer.layer_name: dictionary}
        return layer_list

    elif isinstance(layer, LayerFieldsContainer):
        if len(layer.fields) == 1:
            return add_layer_to_dict(layer.fields[0])
        d_list = []
        for field in layer.fields:
            d_list.append(add_layer_to_dict(field))
        dictionary = defaultdict(list)
        for d in d_list:  # you can list as many input dicts as you want here
            for key, value in d.items():
                dictionary[key].append(value)
        return dictionary

    elif isinstance(layer, LayerField):
        return {layer.name: layer.show}

    elif isinstance(layer, list):  # FIXME no coverage in XML mode -> check with other PCAPs
        d_list = []
        for sub_layer in layer:
            d_list += [add_layer_to_dict(sub_layer)]
        return d_list


def flatten_dict(dictionary, par_key=""):
    seperator = "."
    items = {}
    for key, val in dictionary.items():
        cur_key = par_key + seperator + key if par_key else key
        if isinstance(val, MutableMapping):
            items.update(flatten_dict(val, cur_key))
        else:
            items.update({cur_key: val})
    return items


def dict2json(dictionary):
    return json.dumps(dictionary, indent=2)


def packet2dict(p: Packet):
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
        p_dict.update(add_layer_to_dict(layer))

    return dict2json(flatten_dict(p_dict))


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.DEBUG)
    endpoint = stream.StreamEndpoint(addr=("127.0.0.1", 12000), endpoint_type=stream.SINK,
                                     multithreading=True, buffer_size=10000)
    endpoint.start()

    count = 1
    t_size = 0
    # while True:
    #     packet = endpoint.receive()
    #     packet = typing.cast(Packet, packet)
    for packet in endpoint:
        d_packet = packet2dict(packet)
        t_size += len(pickle.dumps(d_packet))
        logging.info(f"Received Pyshark Packet {count}, total {t_size}")
        count += 1
