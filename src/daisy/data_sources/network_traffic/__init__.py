# Copyright (C) 2024 DAI-Labor and others
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
    * PysharkProcessor - Able to process pyshark packet objects into numpy vectors.

There is also a module specialized for traffic of cohda boxes (V2X), that offers
additional functionalities:

    * march23_events - Event tags for labeling purposes for the March23 dataset.

Author: Fabian Hofmann, Jonathan Ackerschewski, Seraphin Zunzer
Modified: 19.04.24
"""

__all__ = [
    "LivePysharkDataSource",
    "PcapDataSource",
    "create_pyshark_processor",
    "dict_to_numpy_array",
    "packet_to_dict",
    "dict_to_json",
    "default_f_features",
    "march23_events",
    "label_data_point",
    "default_nn_aggregator",
]

from .demo_202312 import default_f_features, march23_events, label_data_point
from .pyshark_handler import LivePysharkDataSource, PcapDataSource
from .pyshark_processor import (
    create_pyshark_processor,
    dict_to_numpy_array,
    packet_to_dict,
    dict_to_json,
    default_nn_aggregator,
)
