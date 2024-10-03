# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""A collection of interfaces and base classes for data stream generation and
preprocessing for further (ML) tasks. The data source module is the core of the
package while modules and further subpackages are implementations/extensions of the
provided interfaces to enable this framework for various use-cases.

    * DataSource - Core class of any data source, union of DataProcessor, SourceHandler.
    * SourceHandler - Interface class. Implementations must provide data point
    objects in generator-like fashion.
    * DataProcessor - Interface class. Implementations must process data point
    objects into vectors (numpy arrays).
    * DataSourceRelay - Second core class that allows the processing and forwarding
    of data points to another host.
    * CSVFileRelay - Third core class that allows the export of data points to CSV.

Currently, the following sub-packages are offering interface implementations:

    * network_traffic - Handling and processing of data points (network packets)
    originating from t-/wireshark or pcaps. See the subpackage documentation for more.

Author: Fabian Hofmann, Jonathan Ackerschewski, Seraphin Zunzer
Modified: 16.04.24
"""

__all__ = [
    "SourceHandler",
    "SimpleSourceHandler",
    "SimpleRemoteSourceHandler",
    "DataProcessor",
    "SimpleDataProcessor",
    "IdentityDataProcessor",
    "remove_filter_fn",
    "DataSourceRelay",
    "CSVFileRelay",
    "DataSource",
    "CohdaProcessor",
    "march23_events",
    "LivePysharkHandler",
    "PcapHandler",
    "pyshark_map_fn",
    "pyshark_filter_fn",
    "pyshark_reduce_fn",
    "create_pyshark_processor",
]

from .data_handler import SourceHandler, SimpleSourceHandler, SimpleRemoteSourceHandler
from .data_processor import (
    DataProcessor,
    SimpleDataProcessor,
    IdentityDataProcessor,
    remove_filter_fn,
)
from .data_relay import DataSourceRelay, CSVFileRelay
from .data_source import DataSource
from .network_traffic import (
    CohdaProcessor,
    march23_events,
    LivePysharkHandler,
    PcapHandler,
    pyshark_map_fn,
    pyshark_filter_fn,
    pyshark_reduce_fn,
    create_pyshark_processor,
)
