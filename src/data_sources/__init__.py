"""TODO
    A collection of interfaces and base classes for data stream generation and preprocessing for further (ML) tasks.
    Supports generic generators, but also remote communication endpoints that hand over generic data points in
    streaming-manner, and any other implementations of the SourceHandler class. Note each different kind of data needs
    its own implementation of the DataProcessor class.

    Author: Fabian Hofmann, Jonathan Ackerschewski
    Modified: 26.07.23
"""

from .data_source import DataSource, DataProcessor, SourceHandler
from .data_source import SimpleSourceHandler, SimpleRemoteSourceHandler

from .network_traffic import *
