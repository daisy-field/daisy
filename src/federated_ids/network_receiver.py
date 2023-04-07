import json
import logging
from collections import defaultdict
from collections.abc import MutableMapping

from pyshark.packet.fields import LayerField
from pyshark.packet.fields import LayerFieldsContainer
from pyshark.packet.layers.xml_layer import XmlLayer
from pyshark.packet.packet import Packet

from communication.message_stream import StreamEndpoint


# TODO further cleanup, docstrings, typehints, splitting it from pyshark capture
# TODO more copy pasta (e.g. argparse)

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
    logging.basicConfig(level=logging.DEBUG)
    endpoint = StreamEndpoint(addr=("127.0.0.1", 12000), multithreading=True)
    endpoint.start(StreamEndpoint.EndpointType.SINK)

    sum = 0
    while True:
        packet = endpoint.receive()
        sum += len(packet)
        d_packet = packet2dict(packet)
        logging.info(f"Received Pyshark Packet: {len(packet)}, total data: {sum}")
