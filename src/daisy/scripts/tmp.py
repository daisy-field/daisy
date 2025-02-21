# Copyright (C) 2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from daisy.data_sources import (
    PcapDataSource,
    PysharkProcessor,
    DataHandler,
)
import time

result = {}


def count(x):
    result[x["label"]] = result.get(x["label"], 0) + 1
    return x


def print_result():
    for key, value in result.items():
        print(f"{key}: {value}")


def cast(x):
    x["meta.time_epoch"] = float(x["meta.time_epoch"])
    return x


def cast2(x):
    x["ip.addr"] = str(x["ip.addr"])
    return x


result = {}
source = PcapDataSource("../../../../datasets/v2x_2023-03-06/diginet-cohda-box-dsrc2/")
processor = (
    PysharkProcessor().packet_to_dict()
    # .select_dict_features(features=pcap_f_features)
    # .add_func(cast)
    # .add_func(cast2)
    # .add_func(lambda x: demo_202303_label_data_point(2, x))
    # .add_func(count)
)
# processor = PysharkProcessor().packet_to_dict().select_dict_features(features=pcap_f_features).merge_dict({"client_id": 2}).add_func(cast).add_func(cast2).add_event_handler(march23_event_handler).add_func(count)
handler = DataHandler(source, processor, multithreading=False)
packets = 0
delta_packets = 0
step = time.time()
handler.open()
with open("packets.txt", "w") as f:
    for x in handler:
        packets += 1
        f.write(str(x) + "\n")
        delta_packets += 1
        if time.time() - step > 1:
            delta = time.time() - step
            step = time.time()
            # print(delta_packets / delta)
            delta_packets = 0
        if packets % 10000 == 0:
            print(f"Cur Packet: {packets}")
        if packets == 250000:
            break
handler.close()
print_result()
