"""
    A collection of interfaces and base classes for data stream generation and preprocessing for further (ML) tasks.
    Supports generic generators, but also remote communication endpoints that hand over generic data points in
    streaming-manner, and any other implementations of the SourceHandler class. Note each different kind of data needs
    its own implementation of the DataProcessor class.

    Author: Fabian Hofmann, Jonathan Ackerschewski
    Modified: 27.07.23
"""

import logging
import queue
import threading
from abc import ABC, abstractmethod
from typing import Callable, Iterator

import numpy as np

from src.communication import StreamEndpoint


# TODO logging: levels, messages, names
# TODO factory functions

class DataProcessor(ABC):
    """An abstract data processor that has to process data points as they come in the following three steps:

        - map(): The object is mapped to a dictionary, that includes all the features from the data point.

        - filter(): Those features are filtered, based on the need of the applications that use the data.

        - reduce(): The data point is reduced and converted into a numpy array, stripping it of all feature names.

    Any implementation has to funnel all its functionalities through these three methods (besides __init__), as they are
    called through process() by the DataSource.
    """

    @abstractmethod
    def map(self, o_point: object) -> dict:
        """Deserializes a data object into a dictionary with the data point's feature names as keys and values as
        values.

        :param o_point: Data point as object.
        :return: Data point as dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def filter(self, d_point: dict) -> dict:
        """Filters the data point dictionary by removing features from the vector based on a set condition (or filter).

        :param d_point: Data point as dictionary.
        :return: Data point as dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def reduce(self, d_point: dict) -> np.ndarray:
        """Reduces the data point dictionary into a numpy array/vector, stripped from any feature names and redundant
        information.

        :param d_point: Data point as dictionary.
        :return: Data point as numpy array.
        """
        raise NotImplementedError

    def process(self, o_point: object) -> np.ndarray:
        """Converts and processes a data point object into a feature vector (numpy array).

        Method called by DataSource objects as they load and process the data stream. Overriding should not be needed
        as all to be implemented functionality should be covered by the three abstract methods that are called by it.

        :param o_point: Data point as object.
        :return: Processed data point as vector.
        """
        return self.reduce(self.filter(self.map(o_point)))


class SimpleDataProcessor(DataProcessor):
    """The simplest productive data processor --- a wrapper around a callable function which directly transforms a given
    data point object into a numpy array, skipping all intermediary steps. Can also be used if the given format of map,
    filter, reduce, does not fit the requirements.
    """
    _process_fn: Callable[[object], np.ndarray]

    def __init__(self, process_fn: Callable[[object], np.ndarray]):
        """Creates a data processor, simply wrapping it around the given callable.

        :param process_fn: Callable object with which data points can be processed.
        """
        self._process_fn = process_fn

    def map(self, o_point: object) -> dict:
        pass

    def filter(self, d_point: dict) -> dict:
        pass

    def reduce(self, d_point: dict) -> np.ndarray:
        pass

    def process(self, o_point: object) -> np.ndarray:
        """Converts and processes a data point object into a feature vector (numpy array), using the wrapped callable.

        :param o_point: Data point as object.
        :return: Processed data point as vector.
        """
        return self._process_fn(o_point)


class SourceHandler(ABC):
    """An abstract wrapper around a generator-like structure that has to yield data points as objects as they come for
    processing. That generator may be infinite or finite, as long as it is bounded on both sides by the following two
    methods that must be implemented:

        - open(): Enables the "generator" to provision data points.

        - close(): Closes the "generator".

    Note that as DataSource, wraps itself a given source handler to retrieve objects, open() and close() do not need to
    be implemented to be idempotent and arbitrarily permutable. Same can be assumed for __iter__() as it will only be
    called when the source handler has been opened already. At the same time, __iter__() must be exhausted after close()
    has been called.
    """

    @abstractmethod
    def __init__(self):
        """Creates the source handler. Note that this should not enable the immediate generation of data points via
        __iter__() --- this behavior is implemented through open() (see the class documentation for more information).
        """
        raise NotImplementedError

    @abstractmethod
    def open(self):
        """Prepares the handler to be used for data point generation, setting up necessary environment variables,
        starting up background processes to read/generate data, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Closes the handler after which data point generation is no longer available until opened again.
        """
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[object]:
        """After opened (see open()), returns a generator - either the object itself or creates a new one (e.g. through
        use of the yield statement).

        :return: Generator object for data points as objects.
        """
        raise NotImplementedError


class SimpleSourceHandler(SourceHandler):
    """The simplest productive source handler --- an actual wrapper around a generator that is always open and cannot be
    closed, yielding data points as objects as they are yielded. Can be infinite or finite; no matter, no control over
    the generator is natively supported.
    """
    _generator: Iterator[object]

    def __init__(self, generator: Iterator[object]):
        """Creates a source handler, simply wrapping it around the given generator.

        :param generator: Generator object from which data points are retrieved.
        """
        self._generator = generator

    def open(self):
        pass

    def close(self):
        pass

    def __iter__(self) -> Iterator[object]:
        """Returns the wrapped generator, requiring neither open() nor close().

        :return: Generator object for data points as objects.
        """
        return self._generator


class SimpleRemoteSourceHandler(SourceHandler):
    """The simple wrapper implementation to support and handle remote streaming endpoints of the Endpoint module as data
    sources. Considered infinite in nature, as it allows the generation of data point objects from a connected
    endpoint, until the client closes the handler.
    """
    _logger: logging.Logger
    _endpoint: StreamEndpoint

    def __init__(self, endpoint: StreamEndpoint = None,
                 addr: tuple[str, int] = ("127.0.0.1", 12000), remote_addr: tuple[str, int] = None,
                 multithreading: bool = False, buffer_size: int = 1024):
        """Creates a new remote source handler from a given stream endpoint. If no endpoint is provided, creates a new
        one instead with basic parameters.

        :param endpoint: Streaming endpoint from which data points are retrieved.
        :param addr: If no endpoint is provided, local address of new endpoint.
        :param multithreading: Enables transparent multithreading for speedup.
        :param buffer_size: Size of shared buffer in multithreading mode.
        """
        self._logger = logging.getLogger()
        self._logger.info("Initializing remote source handler...")
        if endpoint is None:
            endpoint = StreamEndpoint("RemoteSourceHandler", addr, remote_addr, acceptor=True,
                                      multithreading=multithreading, buffer_size=buffer_size)
        self._endpoint = endpoint
        self._logger.info("Remote source handler initialized.")

    def open(self):
        """Starts and opens/connects the endpoint of the source handler.
        """
        self._logger.info("Starting remote data source...")
        try:
            self._endpoint.start()
        except RuntimeError:
            pass
        self._logger.info("Remote data source started.")

    def close(self):
        """Stops and closes the endpoint of the source handler.
        """
        self._logger.info("Stopping remote data source...")
        try:
            self._endpoint.stop()
        except RuntimeError:
            pass
        self._logger.info("Remote data source stopped.")

    def __iter__(self) -> Iterator[object]:
        """Returns the wrapped endpoint generator, as it supports object retrieval directly.

        :return: Endpoint generator object for data points as objects.
        """
        return self._endpoint.__iter__()


class DataSource:
    """A wrapper around a customizable SourceHandler that yields data points as objects as they come, before stream
    processing using another, customizable DataProcessor. Data points, which can be from arbitrary sources, are thus
    processed and converted into numpy vectors/arrays.

    Supports the processing of data points in both synchronous and asynchronous fashion by default.
    """
    _logger: logging.Logger

    _source_handler: SourceHandler
    _data_processor: DataProcessor

    _multithreading: bool
    _thread: threading.Thread
    _buffer: queue.Queue
    _opened: bool

    def __init__(self, source_handler: SourceHandler = None, generator: Iterator[object] = None,
                 data_processor: DataProcessor = None, process_fn: Callable[[object], np.ndarray] = lambda o: o,
                 multithreading: bool = False, buffer_size: int = 1024):
        """Creates a new data source.

        :param source_handler: Actual source that provisions data points to data source.
        :param generator: Generator object from which data points are retrieved, fallback from source handler.
        :param data_processor: Processor containing the methods on how to process individual data points.
        :param process_fn: Processor functioning to process individual data points. If neither processor nor function
        provided, defaults to NOP.
        :param multithreading: Enables transparent multithreading for speedup.
        :param buffer_size: Size of shared buffer in multithreading mode.
        """
        self._logger = logging.getLogger()
        self._logger.info("Initializing data source...")

        if source_handler is not None:
            self._source_handler = source_handler
        elif generator is not None:
            self._source_handler = SimpleSourceHandler(generator)
        else:
            raise ValueError("Data source requires either a data source handler or a generator to load data from!")

        if data_processor is not None:
            self._data_processor = data_processor
        else:
            self._data_processor = SimpleDataProcessor(process_fn)

        self._multithreading = multithreading
        self._buffer = queue.Queue(buffer_size)
        self._opened = False
        self._logger.info("Data source initialized.")

    def open(self):
        """Opens the data source for data point retrieval. Must be called before data can be retrieved; in
        multithreading mode also starts the loader thread as daemon.
        """
        self._logger.info("Starting data source...")
        if self._opened:
            raise RuntimeError(f"Data source has already been opened!")
        self._opened = True
        self._source_handler.open()

        if self._multithreading:
            self._thread = threading.Thread(target=self._loader, daemon=True)
            self._thread.start()
        self._logger.info("Data source started.")

    def close(self):
        """Shuts down any thread running in the background to load data into the data source iff in multithreading mode.
        Can be reopened (and closed) and arbitrary amount of times.
        """
        self._logger.info("Stopping data source...")
        if not self._opened:
            raise RuntimeError(f"Data source has not been opened!")
        self._opened = False
        self._source_handler.close()

        if self._multithreading:
            self._thread.join()
        self._logger.info("Data source stopped.")

    def _loader(self):
        """Data loader for multithreading mode, loads data from source handlers and processes it to store it in the
        shared buffer.
        """
        self._logger.info(f"AsyncLoader: Starting to process data points in asynchronous mode...")
        for o_point in self._source_handler:
            while self._opened:
                try:
                    self._logger.debug(
                        f"AsyncLoader: Storing processed data point in buffer "
                        f"(length: {self._buffer.qsize()})...")
                    self._buffer.put(self._data_processor.process(o_point), timeout=10)
                except queue.Full:
                    self._logger.warning(f"AsyncLoader: Timeout triggered: Buffer full. Retrying...")
            if not self._opened:
                break
        self._logger.info(f"AsyncLoader: Stopping...")

    def __iter__(self) -> Iterator[np.ndarray]:
        """Generator that supports multithreading to retrieve processed data points.

        :return: Generator object for data points as numpy arrays.
        """
        self._logger.info("Retrieving data points from data source...")
        if not self._opened:
            raise RuntimeError("Data source has not been opened!")

        if self._multithreading:
            while self._opened:
                self._logger.debug(
                    f"Multithreading detected, retrieving data point from buffer (size={self._buffer.qsize()})...")
                try:
                    yield self._buffer.get(timeout=10)
                except queue.Empty:
                    self._logger.warning(f"Timeout triggered: Buffer empty. Retrying...")
        else:
            for o_point in self._source_handler:
                yield self._data_processor.process(o_point)
        self._logger.info("Data source exhausted or closed.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        if self._opened:
            self.close()


class DataSourceRelay:
    """A union of a data source and a stream endpoint to retrieve data points from the former and relay them over the
    latter. This allows the disaggregation of the actual datasource from the other processing steps. For example, the
    relay could be deployed with our without an actual processor on another host and the data is forwarded over the
    network to another host running a data source with a SimpleRemoteSourceHandler to receive and further process the
    data. This chain could also be continued beyond a single host pair.
    """
    _logger: logging.Logger

    _data_source: DataSource
    _endpoint: StreamEndpoint

    _thread: threading.Thread
    _started: bool

    def __init__(self, data_source: DataSource = None, endpoint: StreamEndpoint = None,
                 source_handler: SourceHandler = None, generator: Iterator[object] = None,
                 data_processor: DataProcessor = None, process_fn: Callable[[object], np.ndarray] = lambda o: o,
                 addr: tuple[str, int] = ("127.0.0.1", 12000), remote_addr: tuple[str, int] = None,
                 multithreading: bool = False, buffer_size: int = 1024):
        """Creates a new data source relay. If either isn't provided, one can also provide the basic parameters for the
        creation of data source and/or endpoint.

        :param data_source: Data source to relay data points from.
        :param endpoint: Streaming endpoint to which data points are relayed to.
        :param source_handler: Actual source that provisions data points to data source. Fallback from data source.
        :param generator: Generator object from which data points are retrieved, fallback from source handler.
        :param data_processor: Processor containing the methods on how to process individual data points.
        :param process_fn: Processor functioning to process individual data points. If neither processor nor function
        provided, defaults to NOP.
        :param addr: Local address of new endpoint. Fallback from endpoint.
        :param remote_addr: Address of remote endpoint to be connected to. Fallback from endpoint.
        :param multithreading: Enables multithreading of data source and endpoint for speedup.
        :param buffer_size: Size of shared buffers in multithreading mode.
        """
        self._logger = logging.getLogger()

        if data_source is None:
            data_source = DataSource(source_handler, generator, data_processor, process_fn, multithreading, buffer_size)
        self._data_source = data_source

        if endpoint is None:
            endpoint = StreamEndpoint("DataSourceRelay", addr, remote_addr, acceptor=False,
                                      multithreading=multithreading, buffer_size=buffer_size)
        self._endpoint = endpoint

        self._started = False

    def start(self):
        """Starts the data source relay along any other objects in this union (data source, endpoint). Non-blocking, as
        the relay is started in the background to allow the stopping of it afterward.
        """
        self._logger.info("Starting data source relay...")
        if self._started:
            raise RuntimeError(f"Relay has already been started!")
        self._started = True
        try:
            self._data_source.open()
        except RuntimeError:
            pass
        try:
            self._endpoint.start()
        except RuntimeError:
            pass

        self._thread = threading.Thread(target=self._relay, daemon=True)
        self._thread.start()
        self._logger.info("Data source relay started.")

    def stop(self):
        """Closes and stops the data source and the endpoint and joins the relay thread into the current thread. Can be
        restarted (and stopped) and arbitrary amount of times.
        """
        self._logger.info("Stopping data source relay...")
        if not self._started:
            raise RuntimeError(f"Endpoint has not been started!")
        self._started = False
        try:
            self._data_source.close()
        except RuntimeError:
            pass
        try:
            self._endpoint.stop()
        except RuntimeError:
            pass

        self._thread.join()
        self._logger.info("Data source relay stopped.")

    def _relay(self):
        """Actual relay, directly forwards data points from its data source to its endpoint (both might be async).
        """
        self._logger.info("Starting to relay data points from data source...")
        for d_point in self._data_source:
            try:
                self._endpoint.send(d_point)
            except RuntimeError:
                break
        self._logger.info("Data source exhausted, or relay closing...")

    def __del__(self):
        if self._started:
            self.stop()
