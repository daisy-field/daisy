"""
    A collection of interfaces for data stream pre-processing and generation for further (ML) tasks. Supports generic
    underlying generators, but also remote communication endpoints that hand over generic data points in
    streaming-manner.

    Author: Fabian Hofmann
    Modified: 14.04.22
"""

import logging
import queue
import threading
from abc import ABC, abstractmethod
from typing import Iterator, Tuple

import numpy as np

from src.communication.message_stream import StreamEndpoint, SINK


class DataSource(ABC):
    """An abstract wrapper around an existing data point generator that yields data points as objects as they come,
    before stream processing them in three steps that have to be implemented: First the object is MAPped to a
    dictionary, that includes all the features from the data point. Then those features are FILTERed, based on the need
    of the applications that use the data. Finally, the data point is REDUCEd and converted into a numpy array,
    stripping it of all feature names.

    Supports the processing of data points in both synchronous and asynchronous fashion by default.
    """
    _logger: logging.Logger

    _generator: Iterator[object]

    _multithreading: bool
    _thread: threading.Thread
    _buffer: queue.Queue[np.ndarray]
    _opened: bool

    def __init__(self, generator: Iterator[object], multithreading: bool = False, buffer_size: int = 1024):
        """Creates a new data source.

        :param generator: Generator object from which data points are retrieved.
        :param multithreading: Enables transparent multithreading for speedup.
        :param buffer_size: Size of shared buffer in multithreading mode.
        """
        self._logger = logging.getLogger()
        self._logger.info("Initializing data source...")

        self._generator = generator

        self._multithreading = multithreading
        self._buffer = queue.Queue(buffer_size)
        self._opened = False
        self._logger.info("Data source initialized.")

    @abstractmethod
    def map(self, o_point: object) -> dict:
        """Deserializes a data object into a dictionary with the datapoint's feature names as keys and values as values.

        :param o_point: Datapoint as object
        :return: Datapoint as dictionary
        """
        raise NotImplementedError

    @abstractmethod
    def filter(self, d_point: dict) -> dict:
        """Filters the datapoint dictionary by removing features from the vector based on a set condition (or filter).

        :param d_point: Datapoint as dictionary.
        :return: Datapoint as dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def reduce(self, d_point: dict) -> np.ndarray:
        """Reduces the datapoint dictionary into a numpy array/vector, stripped from any feature names and redundant
        information.

        :param d_point: Datapoint as dictionary.
        :return: Datapoint as numpy array.
        """
        raise NotImplementedError

    def open(self):
        """Must be called before data can be retrieved; in multithreading mode also starts the loader thread as daemon.
        """
        self._logger.info("Starting data source...")
        if self._opened:
            raise RuntimeError(f"Data source has already been opened!")
        self._opened = True

        if self._multithreading:
            self._thread = threading.Thread(target=self._loader, name="DataSourceLoader", daemon=True)
            self._thread.start()
        self._logger.info("Data source started.")

    def close(self):
        """Shuts down any thread running in the background to load data into the data source iff in multithreading mode.
        Can be reopened.
        """
        self._logger.info("Stopping data source...")
        if not self._opened:
            raise RuntimeError(f"Data source has not been opened!")
        self._opened = False

        if self._multithreading:
            self._thread.join()
        self._logger.info("Data source stopped.")

    def _process(self, o_point: object) -> np.ndarray:
        return self.reduce(self.filter(self.map(o_point)))

    def _loader(self):
        """Data loader for multithreading mode, loads data from the generator and processes it to store it in the shared
        buffer.
        """
        self._logger.info(f"{self._thread.name}: Starting to process data points in asynchronous mode...")
        for o_point in self._generator:
            while self._opened:
                try:
                    self._logger.debug(
                        f"{self._thread.name}: Storing processed datapoint in buffer "
                        f"(length: {self._buffer.qsize()})...")
                    self._buffer.put(self._process(o_point), timeout=10)
                except queue.Full:
                    self._logger.warning(f"{self._thread.name}: Timeout triggered: Buffer full. Retrying...")
            if not self._opened:
                break
        self._logger.info(f"{self._thread.name}: Stopping...")

    def __iter__(self) -> Iterator[np.ndarray]:
        """Generator that supports multithreading to retrieve processed datapoints.

        :return: Generator object for datapoints as numpy arrays.
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
            for o_point in self._generator:
                yield self._process(o_point)
        self._logger.info("Data source exhausted or closed.")

    def __del__(self):
        if self._opened:
            self.close()


class RemoteDataSource(DataSource, ABC):
    """An abstract implementation of a data source, that comes with support to the streaming endpoint module, as it
    allows the generation of data point objects from a connected endpoint.
    """
    _endpoint: StreamEndpoint

    def __init__(self, endpoint: StreamEndpoint = None, addr: Tuple[str, int] = ("127.0.0.1", 12000),
                 multithreading: bool = False, buffer_size: int = 1024):
        """Creates a new remote data source from a given stream endpoint. If no endpoint is provided, creates a new one
        instead with basic parameters.

        :param endpoint: Streaming endpoint from which data points are retrieved.
        :param addr: If no endpoint is provided, local address of new endpoint.
        :param multithreading: Enables transparent multithreading for speedup.
        :param buffer_size: Size of shared buffer(s) in multithreading mode.
        """
        self._logger = logging.getLogger()
        self._logger.info("Initializing remote data source...")
        if endpoint is None:
            endpoint = StreamEndpoint(addr=addr, endpoint_type=SINK,
                                      multithreading=multithreading, buffer_size=buffer_size)
        self._endpoint = endpoint

        super().__init__(self._endpoint.__iter__(), multithreading, buffer_size)
        self._logger.info("Remote data source initialized.")

    def open(self):
        """Like the super-method, but also starts and opens/connects the endpoint.
        """
        self._logger.info("Starting remote data source...")
        try:
            self._endpoint.start()
        except RuntimeError:
            pass
        super().open()
        self._logger.info("Remote data source started.")

    def close(self):
        """Like the super-method, but also stops and closes the endpoint.
        """
        self._logger.info("Stopping remote data source...")
        try:
            self._endpoint.stop()
        except RuntimeError:
            pass
        super().close()
        self._logger.info("Remote data source stopped.")

    def __del__(self):
        if self._opened:
            self.close()
