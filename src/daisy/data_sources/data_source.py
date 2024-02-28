"""
    A collection of interfaces and base classes for data stream generation and preprocessing for further (ML) tasks.
    Supports generic generators, but also remote communication endpoints that hand over generic data points in
    streaming-manner, and any other implementations of the SourceHandler class. Note each different kind of data needs
    its own implementation of the DataProcessor class.

    Author: Fabian Hofmann, Jonathan Ackerschewski
    Modified: 27.07.23

    TODO Future Work: Defining granularity of logging in inits
    TODO Future Work: Cleanup of inits to eliminate overlap of classes
"""

import logging
import queue
import threading
from typing import Iterator

import numpy as np

from daisy.data_sources import SourceHandler, DataProcessor


class DataSource:  # TODO comments and variable declarations
    """A wrapper around a customizable SourceHandler that yields data points as objects as they come, before stream
    processing using another, customizable DataProcessor. Data points, which can be from arbitrary sources, are thus
    processed and converted into numpy vectors/arrays.

    Supports the processing of data points in both synchronous and asynchronous fashion by default.
    """
    _logger: logging.Logger

    _source_handler: SourceHandler
    _data_processor: DataProcessor

    _multithreading: bool
    _loader: threading.Thread
    _buffer: queue.Queue
    _opened: bool

    def __init__(self, source_handler: SourceHandler, data_processor: DataProcessor, name: str = "",
                 multithreading: bool = False, buffer_size: int = 1024):
        """Creates a new data source.

        :param name: Name of data source relay for logging purposes.
        :param source_handler: Actual source that provisions data points to data source.
        :param generator: Generator object from which data points are retrieved, fallback from source handler.
        :param data_processor: Processor containing the methods on how to process individual data points.
        :param process_fn: Processor function to process individual data points. If neither processor nor function
        provided, defaults to NOP.
        :param multithreading: Enables transparent multithreading for speedup.
        :param buffer_size: Size of shared buffer in multithreading mode.
        """
        self._logger = logging.getLogger(name)
        self._logger.info("Initializing data source...")

        self._opened = False

        self._source_handler = source_handler
        self._data_processor = data_processor

        self._multithreading = multithreading
        self._buffer = queue.Queue(buffer_size)
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
            self._loader = threading.Thread(target=self._create_loader, daemon=True)
            self._loader.start()
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
            self._loader.join()
        self._logger.info("Data source stopped.")

    def _create_loader(self):
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
