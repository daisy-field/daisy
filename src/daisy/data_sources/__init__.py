# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""TODO REVIEW comments
A collection of interfaces and base classes for data stream generation and preprocessing for further (ML) tasks.
The data source module is the core of the package while further subpackages are implementations/extensions of the
provided interfaces to enable this framework for various use-cases.

    * DataSource - Core class of the data source framework, union of DataProcessor, SourceHandler.
    * SourceHandler - Interface class. Implementations must provide data point objects in generator-like fashion.
    * DataProcessor - Interface class. Implementations must process data point objects into vectors (numpy arrays).
    * DataSourceRelay - Second core class that allows the processing and forwarding of data points to another host.

Currently, the following sub-packages are offering implementations of the two interfaces:

    * network_traffic - Handling and processing of data points (network packets) originating from t-/wireshark.

Author: Fabian Hofmann, Jonathan Ackerschewski, Seraphin Zunzer
Modified: 28.02.24
"""

from .data_handler import SourceHandler, SimpleSourceHandler, SimpleRemoteSourceHandler
from .data_processor import DataProcessor, SimpleDataProcessor
from .data_relay import DataSourceRelay, CSVFileRelay
from .data_source import DataSource
from .network_traffic import *
