# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
A collection of interfaces and base classes for data stream generation and preprocessing for further (ML) tasks.
Supports generic generators, but also remote communication endpoints that hand over generic data points in
streaming-manner, and any other implementations of the SourceHandler class. Note each different kind of data needs
its own implementation of the DataProcessor class.

TODO REFVIEW COMENTS @Fabian
TODO interfacing?

Author: Fabian Hofmann, Jonathan Ackerschewski
Modified: 28.07.23

TODO Future Work: Defining granularity of logging in inits
TODO Future Work: Cleanup of inits to eliminate overlap of classes
"""

import logging
import threading
from pathlib import Path

from daisy.communication import StreamEndpoint
from daisy.data_sources import DataSource


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

    _relay: threading.Thread
    _started: bool

    def __init__(
        self, data_source: DataSource, endpoint: StreamEndpoint, name: str = ""
    ):
        """Creates a new data source relay.

        :param data_source: Data source to relay data points from.
        :param endpoint: Streaming endpoint to which data points are relayed to.
        :param name: Name of data source relay for logging purposes.
        """
        self._logger = logging.getLogger(name)
        self._logger.info("Initializing data source relay...")

        self._started = False

        self._data_source = data_source
        self._endpoint = endpoint

        self._logger.info("Data source relay initialized.")

    def start(self):
        """Starts the data source relay along any other objects in this union (data source, endpoint). Non-blocking, as
        the relay is started in the background to allow the stopping of it afterward.
        """
        self._logger.info("Starting data source relay...")
        if self._started:
            raise RuntimeError("Relay has already been started!")
        self._started = True
        try:
            self._data_source.open()
        except RuntimeError:
            pass
        try:
            self._endpoint.start()
        except RuntimeError:
            pass

        self._relay = threading.Thread(target=self._create_relay, daemon=True)
        self._relay.start()
        self._logger.info("Data source relay started.")

    def stop(self):
        """Closes and stops the data source and the endpoint and joins the relay thread into the current thread. Can be
        restarted (and stopped) and arbitrary amount of times.
        """
        self._logger.info("Stopping data source relay...")
        if not self._started:
            raise RuntimeError("Endpoint has not been started!")
        self._started = False
        try:
            self._data_source.close()
        except RuntimeError:
            pass
        try:
            self._endpoint.stop()
        except RuntimeError:
            pass

        self._relay.join()
        self._logger.info("Data source relay stopped.")

    def _create_relay(self):
        """Actual relay, directly forwards data points from its data source to its endpoint (both might be async)."""
        self._logger.info("Starting to relay data points from data source...")
        for d_point in self._data_source:
            try:
                self._endpoint.send(d_point)
            except RuntimeError:
                # stop() was called
                break
        self._logger.info("Data source exhausted, or relay closing...")

    def __del__(self):
        if self._started:
            self.stop()


class CSVFileRelay:
    """A relay allowing data points to be stored in a CSV file. For this to work, the processor is expected to return
    a dictionary containing values for all fields in the headers parameter.
    """

    _logger: logging.Logger

    _data_source: DataSource
    _file: Path
    _headers: tuple[str, ...]
    _separator: str

    _relay: threading.Thread
    _started: bool

    def __init__(
        self,
        target_file: str,
        data_source: DataSource,
        headers: tuple[str, ...],
        name: str = "",
        overwrite_file: bool = False,
        separator: str = ",",
    ):
        """Creates a new CSV relay instance

        :param target_file: The path to the CSV file. The parent directories will be created if not existent.
        :param data_source: The data source providing the data to store. The processor used by the data source is
        expected to return a dictionary containing all values for all fields in the headers parameter
        :param headers: The headers of the CSV file. The order is preserved in the CSV file
        :param name: Name of the relay for logging purposes
        :param overwrite_file: Whether the CSV file should be overwritten if it exists
        :param separator: The separator used in the CSV file
        """
        self._logger = logging.getLogger(name)
        self._logger.info("Initializing file relay...")

        self._started = False

        if target_file is None or not target_file:
            raise ValueError("File to write to required.")
        self._file = Path(target_file)
        if self._file.is_dir():
            raise ValueError("File path points to a directory instead of a file.")
        # create parent directories, then touch the file, to check whether it exists and is a valid path
        parent_dir = Path(*self._file.parts[:-1])
        parent_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._file.touch(exist_ok=False)
        except FileNotFoundError:
            raise ValueError("File points at an invalid path.")
        except FileExistsError:
            if not overwrite_file:
                raise ValueError("File already exists and should not be overwritten.")

        self._data_source = data_source
        self._separator = separator
        self._headers = headers

        self._logger.info("File relay initialized.")

    def start(self):
        """Starts the data source relay along any other objects in this union. Non-blocking, as
        the relay is started in the background to allow the stopping of it afterward.
        """
        self._logger.info("Starting file relay...")
        if self._started:
            raise RuntimeError("Relay has already been started!")
        self._started = True
        try:
            self._data_source.open()
        except RuntimeError:
            pass

        self._relay = threading.Thread(target=self._create_relay, daemon=True)
        self._relay.start()
        self._logger.info("File relay started.")

    def stop(self):
        """Closes and stops the data source and joins the relay thread into the current thread. Can be
        restarted (and stopped) and arbitrary amount of times.
        """
        self._logger.info("Stopping file relay...")
        if not self._started:
            raise RuntimeError("Relay has not been started!")
        self._started = False
        try:
            self._data_source.close()
        except RuntimeError:
            pass

        self._relay.join()
        self._logger.info("File relay stopped.")

    def _create_relay(self):
        """Writes the data points to a file in a csv style"""
        self._logger.info("Starting to relay data points from data source...")
        with open(self._file, "w") as file:
            file.write(f"{self._separator.join(self._headers)}\n")
            for d_point in self._data_source:
                try:
                    values = map(lambda topic: str(d_point[topic]), self._headers)
                    line = self._separator.join(values)
                    file.write(f"{line}\n")
                except RuntimeError:
                    # stop() was called
                    break
        self._logger.info("Data source exhausted, or relay closing...")

    def __del__(self):
        if self._started:
            self.stop()
