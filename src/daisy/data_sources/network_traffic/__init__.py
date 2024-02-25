# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
    Implementations of the data source helper interface that allows the processing and provisioning of pyshark packets,
    either via file inputs, live capture, or a remote source that generates packets in either fashion.

        * LivePysharkHandler - SourceHandler which simply yields captured packets from a list of interfaces.
        * PcapHandler - SourceHandler which is able to load pcap files sequentially and yield their packets.
        * PysharkProcessor - Able to process pyshark packet objects into numpy vectors.

    There is also a module specialized for traffic of cohda boxes (V2X), that offers additional functionalities:

        * CohdaProcessor - Extension of PysharkProcessor that also supports the labeling of data points.
        * march23_events - Event tags for labeling purposes for the March23 dataset.

    Author: Fabian Hofmann, Jonathan Ackerschewski, Seraphin Zunzer
    Modified: 04.08.23
"""

from .cohda_source import CohdaProcessor
from .cohda_source import march23_events

from .pyshark_source import LivePysharkHandler, PcapHandler
from .pyshark_source import PysharkProcessor
