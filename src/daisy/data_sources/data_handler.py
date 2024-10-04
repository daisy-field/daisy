# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""A collection of the core interface and base classes for the first component of any
data source (see the docstring of the data source class), that provides the origin of
any data points being processed for further (ML) tasks. Supports generic generators,
but also remote communication endpoints that hand over generic data points in
streaming-manner, and any other implementations of the SourceHandler class. Note each
different kind of data may need its own implementation of the SourceHandler.

Author: Fabian Hofmann, Jonathan Ackerschewski
Modified: 16.04.24
"""
# TODO Future Work: Defining granularity of logging in inits

import logging
from abc import ABC, abstractmethod
from io import TextIOWrapper
from typing import Iterator
import csv

from daisy.communication import StreamEndpoint


class SourceHandler(ABC):
    """An abstract wrapper around a generator-like structure that has to yield data
    points as objects as they come for processing. That generator may be infinite or
    finite, as long as it is bounded on both sides by the following two methods that
    must be implemented:

        - open(): Enables the "generator" to provision data points.

        - close(): Closes the "generator".

    Note that as DataSource wraps itself around given source handler to retrieve
    objects, open() and close() do not need to be implemented to be idempotent and
    arbitrarily permutable. Same can be assumed for __iter__() as it will only be
    called when the source handler has been opened already. At the same time,
    __iter__() must be exhausted after close() has been called.
    """

    _logger: logging.Logger

    def __init__(self, name: str = ""):
        """Creates a source handler. Note that this should not enable the immediate
        generation of data points via __iter__() --- this behavior is implemented
        through open() (see the class documentation for more information).

        :param name: Name of handler for logging purposes.
        """
        self._logger = logging.getLogger(name)

    @abstractmethod
    def open(self):
        """Prepares the handler to be used for data point generation, setting up
        necessary environment variables, starting up background processes to
        read/generate data, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Closes the handler after which data point generation is no longer
        available until opened again.
        """
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[object]:
        """After opened (see open()), returns a generator - either the object itself
        or creates a new one (e.g. through use of the yield statement).

        :return: Generator object for data points as objects.
        """
        raise NotImplementedError


class SimpleSourceHandler(SourceHandler):
    """The simplest productive source handler --- an actual wrapper around a
    generator that is always open and cannot be closed, yielding data points as
    objects as they are yielded. Can be infinite or finite; no matter, no control
    over the generator is natively supported.
    """

    _generator: Iterator[object]

    def __init__(self, generator: Iterator[object], name: str = ""):
        """Creates a source handler, simply wrapping it around the given generator.

        :param generator: Generator object from which data points are retrieved.
        :param name: Name of handler for logging purposes.
        """
        super().__init__(name)

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
    """The simple wrapper implementation to support and handle remote streaming
    endpoints of the Endpoint module as data sources. Considered infinite in nature,
    as it allows the generation of data point objects from a connected endpoint,
    until the client closes the handler.
    """

    _endpoint: StreamEndpoint

    def __init__(self, endpoint: StreamEndpoint, name: str = ""):
        """Creates a new remote source handler from a given stream endpoint. If no
        endpoint is provided, creates a new one instead with basic parameters.

        :param endpoint: Streaming endpoint from which data points are retrieved.
        :param name: Name of handler for logging purposes.
        """
        super().__init__(name)

        self._logger.info("Initializing remote source handler...")
        self._endpoint = endpoint
        self._logger.info("Remote source handler initialized.")

    def open(self):
        """Starts and opens/connects the endpoint of the source handler."""
        self._logger.info("Starting remote data source...")
        try:
            self._endpoint.start()
        except RuntimeError:
            pass
        self._logger.info("Remote data source started.")

    def close(self):
        """Stops and closes the endpoint of the source handler."""
        self._logger.info("Stopping remote data source...")
        try:
            self._endpoint.stop()
        except RuntimeError:
            pass
        self._logger.info("Remote data source stopped.")

    def __iter__(self) -> Iterator[object]:
        """Returns the wrapped endpoint generator, as it supports object retrieval
        directly.

        :return: Endpoint generator object for data points as objects.
        """
        return self._endpoint.__iter__()


class CSVFileSourceHandler(SourceHandler):
    """This implementation of the SourceHandler reads one or multiple CSV files and yields their content.
    The output of this class are dictionaries containing the headers (first row) of the CSV files as the keys
    and the line as the values. Each CSV is, therefore, expected to have a header line as the first row.
    """

    _files: str | list[str]
    _is_file_list: bool
    _cur_index: int
    _cur_handle: TextIOWrapper | None
    _cur_csv: csv.reader
    _cur_headers: list[str]

    def __init__(self, files: str | list[str], name: str = ""):
        """Creates an instance of the CSVFileSourceHandler class. Either a single file or a list of files are
        expected as the input.

        :param files: Either a single CSV file or a list of CSV files to read.
        """
        super().__init__(name)

        self._logger.info("Initializing CSV file handler...")
        self._files = files
        self._cur_handle = None
        if isinstance(files, str):
            self._is_file_list = False
        if isinstance(files, list):
            self._is_file_list = True
        self._logger.info("CSV file handler initialized.")

    def open(self):
        """Starts the CSVFileSourceHandler by setting required parameters."""
        self._logger.info("Opening CSV file handler...")
        self._cur_index = -1

    def close(self):
        """Closes the CSVFileSourceHandler."""
        self._logger.info("Closing CSV file handler...")
        if self._cur_handle:
            self._cur_handle.close()
            self._cur_handle = None

    def _open_next_file(self):
        """Opens the next CSV file to read. First, the last read file is closed. Afterwards, the next CSV file is
        opened and the headers are extracted."""
        self._logger.info("Opening next CSV file...")
        if self._cur_handle:
            self._cur_handle.close()
            self._cur_handle = None
        self._cur_index += 1

        if self._is_file_list:
            next_file = self._files[self._cur_index]
        else:
            next_file = self._files

        self._cur_handle = open(next_file, "r")
        self._cur_csv = csv.reader(self._cur_handle)

        self._cur_headers = next(self._cur_csv)
        self._logger.info("Next CSV file opened and headers extracted.")

    @staticmethod
    def _line_to_dict(line, header) -> dict[str, object]:
        """Converts a line into a dictionary using the provided headers.

        :param line: The line to convert.
        :param header: The headers to use as the keys.
        :return: A dictionary containing the headers as keys and the line as values."""
        cur_dict = {}
        if len(line) != len(header):
            pass
        for header_counter in range(len(header)):
            cur_dict[header[header_counter]] = line[header_counter]
        return cur_dict

    def __iter__(self) -> Iterator[object]:
        """Iterates through provided CSV files and yields each line as a dictionary."""
        while (not self._is_file_list and self._cur_index < 0) or (
            self._is_file_list and self._cur_index < len(self._files) - 1
        ):
            self._open_next_file()
            for line in self._cur_csv:
                cur_dict = self._line_to_dict(line, self._cur_headers)
                yield cur_dict
        self._logger.info("All CSV files exhausted.")
