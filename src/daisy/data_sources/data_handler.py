# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Core class wrapper component for stream processing both finite and infinite data
handler into sample-wise data points, each being passed to further (ML) tasks once
and in order. For this a data source is required (see the docstring of the data
source module), that provides the origin of any data points being processed for
further (ML) tasks, and a data processor that prepares the data samples by applying
processing steps to them (see the docstring of the data source module).

Author: Fabian Hofmann, Jonathan Ackerschewski
Modified: 04.11.24
"""

import logging
import queue
import threading
from typing import Iterator

import numpy as np

from .data_processor import DataProcessor
from .data_source import DataSource


class DataHandler:
    """A wrapper around a customizable data source that yields data points as
    objects as they come, before stream processing using another, customizable
    data processor. Data points, which can be from arbitrary sources, are thus
    processed and converted into numpy vectors/arrays for ML tasks. Note that there
    is also the option to keep the object/dict format in case stream processing.

    Supports the processing of data points in both synchronous and asynchronous
    fashion by default.
    """

    _logger: logging.Logger

    _data_source: DataSource
    _data_processor: DataProcessor

    _multithreading: bool
    _loader: threading.Thread
    _buffer: queue.Queue

    _opened: bool
    _exhausted: bool
    _completed = threading.Event

    def __init__(
        self,
        data_source: DataSource,
        data_processor: DataProcessor,
        name: str = "",
        multithreading: bool = False,
        buffer_size: int = 1024,
    ):
        """Creates a new data handler.

        :param data_source: Actual source that provisions data points to data handler.
        :param data_processor: Processor containing the methods on how to process
        individual data points.
        :param name: Name of data source relay for logging purposes.
        :param multithreading: Enables transparent multithreading for speedup.
        :param buffer_size: Size of shared buffer in multithreading mode.
        """
        self._logger = logging.getLogger(name)
        self._logger.info("Initializing data handler...")

        self._opened = False
        self._exhausted = False
        self._completed = threading.Event()

        self._data_source = data_source
        self._data_processor = data_processor

        self._multithreading = multithreading
        self._buffer = queue.Queue(buffer_size)
        self._logger.info("Data handler initialized.")

    def open(self):
        """Opens the data handler for data point retrieval. Must be called before data
        can be retrieved; in multithreading mode also starts the loader thread as
        daemon.

        :return: Event object to check whether data handler has completed processing
        every data point and may be closed. Only useful when iterating through a source
        manually since __iter__() automatically stops yielding objects when completed.
        """
        self._logger.info("Starting data handler...")
        if self._opened:
            raise RuntimeError("Data handler has already been opened!")
        self._opened = True
        self._exhausted = False
        self._completed.clear()
        self._data_source.open()

        if self._multithreading:
            self._loader = threading.Thread(target=self._create_loader, daemon=True)
            self._loader.start()
        self._logger.info("Data handler started.")
        return self._completed

    def close(self):
        """Shuts down any thread running in the background to load data into the data
        handler iff in multithreading mode. Can be reopened (and closed) and arbitrary
        amount of times.
        """
        self._logger.info("Stopping data handler...")
        if not self._opened:
            raise RuntimeError("Data handler has not been opened!")
        self._opened = False
        self._data_source.close()

        if self._multithreading:
            self._loader.join()
        self._logger.info("Data handler stopped.")

    def _create_loader(self):
        """Data loader for multithreading mode, loads data from data source and
        processes it to store it in the shared buffer.
        """
        self._logger.info(
            "AsyncLoader: Starting to process data points in asynchronous mode..."
        )
        for o_point in self._data_source:
            while self._opened:
                try:
                    self._logger.debug(
                        f"AsyncLoader: Storing processed data point in buffer "
                        f"(length: {self._buffer.qsize()})..."
                    )
                    self._buffer.put(self._data_processor.process(o_point), timeout=10)
                    break
                except queue.Full:
                    self._logger.warning(
                        "AsyncLoader: Timeout triggered: Buffer full. Retrying..."
                    )
            if not self._opened:
                break
        if self._opened:
            self._exhausted = True
            self._logger.info("AsyncLoader: Data source exhausted, stopping...")
        self._logger.info("AsyncLoader: Stopped")

    def __iter__(self) -> Iterator[np.ndarray | dict | object]:
        """Generator that supports multithreading to retrieve processed data points.

        :return: Generator object for data points as numpy arrays. Note that for some
        use cases, data processor might keep the object or dictionary structure.
        """
        self._logger.info("Retrieving data points from data handler...")
        if not self._opened:
            raise RuntimeError("Data handler has not been opened!")

        if self._multithreading:
            while self._opened and not (self._buffer.empty() and self._exhausted):
                self._logger.debug(
                    "Multithreading detected, retrieving data point from "
                    f"buffer (size={self._buffer.qsize()})..."
                )
                try:
                    yield self._buffer.get(timeout=10)
                except queue.Empty:
                    self._logger.warning("Timeout triggered: Buffer empty. Retrying...")
        else:
            for o_point in self._data_source:
                yield self._data_processor.process(o_point)
        self._logger.info("Data source exhausted or closed.")
        self._completed.set()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        if self._opened and threading.current_thread() != self._loader:
            self.close()
