import json
import pyshark
from pyshark.packet.packet import Packet

from collections import defaultdict
import argparse
import datetime
import os
import re
from collections.abc import MutableMapping
from time import time

import pandas as pd

def addLayerToDict(layer):
    if isinstance(layer, pyshark.packet.layers.xml_layer.XmlLayer):
        dict = {}
        for field in layer.field_names:
            result = addLayerToDict(layer.get_field(field))
            if isinstance(result, list):
                if hasattr(layer.get_field(field), "layer_name"):
                    dict.update({layer.get_field(field).layer_name: result})
                else:
                    # dict.update({f"RandomFieldName{randint(100000, 999999)}": result})
                    # Assumptions:
                    # - result has no layer_name, because it is a list
                    # - the list is never empty
                    # - all elements inside the list are dictionaries
                    # - all dictionaries inside the list have one element
                    # - all dictionaries inside the list share the same key
                    # If this fails, try enabling the RandomFieldName above and look in the generated json
                    # what the result looks like and which assumption doesn't hold
                    dict.update({next(iter(result[0].keys())): [res[next(iter(result[0].keys()))] for res in result]})
            else:
                dict.update(result)
        layer_list = {layer.layer_name: dict}

        return layer_list
    elif isinstance(layer, pyshark.packet.fields.LayerFieldsContainer):
        if len(layer.all_fields) > 5:
            print(f"Layer count: {len(layer.all_fields)}, layer: {layer.all_fields}")
        d_list = []
        for field in layer.all_fields:
            d_list.append(addLayerToDict(field))
        dict = defaultdict(list)
        for d in d_list: # you can list as many input dicts as you want here
            for key, value in d.items():
                dict[key].append(value)
        return dict
    elif isinstance(layer, pyshark.packet.fields.LayerField):
        return {layer.name: layer.show}
    elif isinstance(layer, list):
        l = []
        for sub_layer in layer:
            l += [addLayerToDict(sub_layer)]

        return l

def packet2dict(p: Packet):
    packet_dict = {}

    meta_dict = {
        "number": p.number,
        "len": p.length,
        "protocols": [x.layer_name for x in p.layers],
        "time_epoch": p.sniff_timestamp,
        "time": str(p.sniff_time)
    }
    packet_dict.update({"meta": meta_dict})

    for layer in p.layers:
        packet_dict.update(addLayerToDict(layer))

    return packet_dict


def pyshark_capture():
    capture = pyshark.LiveCapture(interface='any')
    p: Packet
    for p in capture.sniff_continuously():
        p_dict = packet2dict(p)
        print(p_dict)

if __name__ == '__main__':
    pyshark_capture()
