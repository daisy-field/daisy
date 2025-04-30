# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""A collection of interfaces and base classes for data stream generation and
preprocessing for further (ML) tasks. The data source module is the core of the
package while modules and further subpackages are implementations/extensions of the
provided interfaces to enable this framework for various use-cases.

    * DataHandler - Core class of any data handler, union of DataProcessor, DataSource.
    * DataSource - Interface class. Implementations must provide data point
    objects in generator-like fashion.
    * DataProcessor - Processor for data points, applying generic processing steps
    in order to each sample.
    * DataHandlerRelay - Second core class that allows the processing and forwarding
    of data points to another host.
    * CSVFileRelay - Third core class that allows the export of data points to CSV.
    * CSVFileDataHandler - Allows import of data points from CSV files.
    * EventHandler - Provides functionality for labeling data streams automatically.

Currently, the following sub-packages are offering interface implementations:

    * network_traffic - Handling and processing of data points (network packets)
    originating from t-/wireshark or pcaps. See the subpackage documentation for more.

Author: Fabian Hofmann, Jonathan Ackerschewski, Seraphin Zunzer
Modified: 04.11.2024
"""

__all__ = [
    "DataSource",
    "SimpleDataSource",
    "SimpleRemoteDataSource",
    "CSVFileDataSource",
    "DataProcessor",
    "DataHandlerRelay",
    "CSVFileRelay",
    "DataHandler",
    "LivePysharkDataSource",
    "PcapDataSource",
    "PysharkProcessor",
    "pcap_f_features",
    "march23_event_handler",
    "pcap_nn_aggregator",
    "EventHandler",
]

from .data_handler import DataHandler
from .data_processor import (
    DataProcessor,
)
from .data_relay import DataHandlerRelay, CSVFileRelay
from .data_source import (
    DataSource,
    SimpleDataSource,
    SimpleRemoteDataSource,
    CSVFileDataSource,
)
from .events import EventHandler
from .network_traffic import (
    LivePysharkDataSource,
    PcapDataSource,
    PysharkProcessor,
    pcap_f_features,
    march23_event_handler,
    pcap_nn_aggregator,
)
