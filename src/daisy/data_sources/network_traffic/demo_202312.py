# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Content used for the Dataset Demo from 12.2023.

Author: Jonathan Ackerschewski, Fabian Hofmann
Modified: 18.10.2024
"""

from datetime import datetime

import numpy as np

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

# Existing datasets captured on Cohda boxes 2 and 5 on March 6th (2023)
# contains attacks in the following:
# 1: "Installation Attack Tool"
# 2: "SSH Brute Force"
# 3: "SSH Privilege Escalation"
# 4: "SSH Brute Force Response"
# 5: "SSH Data Leakage"
march23_events: list[
    tuple[int, tuple[datetime, datetime], list[str], list[str], int]
] = [
    (
        5,
        (datetime(2023, 3, 6, 12, 34, 17), datetime(2023, 3, 6, 12, 40, 28)),
        ["http", "tcp"],
        ["192.168.213.86", "185."],
        1,
    ),
    (
        5,
        (datetime(2023, 3, 6, 12, 49, 4), datetime(2023, 3, 6, 13, 23, 16)),
        ["ssh", "tcp"],
        ["192.168.230.3", "192.168.213.86"],
        2,
    ),
    (
        5,
        (datetime(2023, 3, 6, 13, 25, 27), datetime(2023, 3, 6, 13, 31, 11)),
        ["ssh", "tcp"],
        ["192.168.230.3", "192.168.213.86"],
        3,
    ),
    (
        2,
        (datetime(2023, 3, 6, 12, 49, 4), datetime(2023, 3, 6, 13, 23, 16)),
        ["ssh", "tcp"],
        ["192.168.230.3", "130.149.98.119"],
        4,
    ),
    (
        2,
        (datetime(2023, 3, 6, 13, 25, 27), datetime(2023, 3, 6, 13, 31, 11)),
        ["ssh", "tcp"],
        ["192.168.230.3", "130.149.98.119"],
        5,
    ),
]


def label_data_point(
    client_id: int,
    events: list[tuple[int, tuple[datetime, datetime], list[str], list[str], int]],
    d_point: dict,
) -> dict:
    """Labels the events according to the provided events.

    :param client_id: The client ID.
    :param events: A list of tuples containing the client ID and timestamp.
    :param d_point: Data point as dictionary.
    :return: Labeled data point as vector.
    """
    d_point["label"] = 0
    for event in events:
        client, (start_time, end_time), protocols, addresses, label = event
        if (
            client == client_id
            and start_time
            <= datetime.strptime(d_point["meta.time"], "%Y-%m-%d %H:%M:%S.%f")
            <= end_time
            and any([x in d_point["meta.protocols"] for x in protocols])
            and d_point["ip.addr"] is not np.nan
            and all([x in d_point["ip.addr"] for x in addresses])
        ):
            d_point["label"] = label
            break
    return d_point
