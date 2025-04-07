# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Implementations of the data handler helper interface that allows the processing and
provisioning of pyshark packets, either via file inputs, live capture, or a remote
source that generates packets in either fashion.

    * LivePysharkDataSource - DataSource which simply yields captured packets from a
    list of interfaces.
    * PcapDataSource - DataSource which is able to load pcap files sequentially and
    yield their packets.
    * PysharkProcessor - Offers additional processing step options to process pyshark
    packet objects.

There is also a module specialized for traffic of cohda boxes (V2X), that offers
additional functionalities:

    * demo_202303- Event tags for labeling purposes for the March23 dataset.

Author: Fabian Hofmann, Jonathan Ackerschewski, Seraphin Zunzer
Modified: 04.11.24
"""

__all__ = [
    "LivePysharkDataSource",
    "PcapDataSource",
    "PysharkProcessor",
    "create_pyshark_processor",
    "dict_to_numpy_array",
    "packet_to_dict",
    "dict_to_json",
    "pcap_f_features",
    "pcap_nn_aggregator",
    "demo_202303_label_data_point",
    "march23_event_handler",
]

from .demo_202303 import demo_202303_label_data_point, march23_event_handler
from .pyshark_handler import LivePysharkDataSource, PcapDataSource
from .pyshark_processor import (
    create_pyshark_processor,
    dict_to_numpy_array,
    packet_to_dict,
    dict_to_json,
    PysharkProcessor,
    pcap_f_features,
    pcap_nn_aggregator,
)
