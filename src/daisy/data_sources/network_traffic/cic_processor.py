# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Implementation of the data processor for supporting processing steps used for pyshark
packets, i.e. a pre-packaged extension of the data processor base class for ease of use.

Author: Jonathan Ackerschewski, Fabian Hofmann
Modified: 04.11.2024
"""

import ipaddress
import sys
from ipaddress import AddressValueError


# Exemplary network feature filter, supporting cohda-box (V2x) messages, besides
# TCP/IP and ETH.
pcap_f_features = (
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


def csv_nn_aggregator(self, key: str, value: object) -> int:
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
            return int(value, 16)
        except ValueError:
            pass
        return hash(value)
    raise ValueError(f"Unable to aggregate non-numerical item: {key, value}")
