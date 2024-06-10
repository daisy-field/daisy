# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""A number of useful tools tht build on top of the data source module, to provide
relays of data points, either over a network over communication endpoints or directly
to a local file(s) on disk. Both wrap around DataSource and thus process the data
stream as it yields data points. Can be used for arbitrarily large arbitrary data
streams.

Author: Fabian Hofmann, Jonathan Ackerschewski
Modified: 17.04.23
"""
# TODO Future Work: Defining granularity of logging in inits

import logging
import threading
from collections import OrderedDict
from pathlib import Path

from daisy.communication import StreamEndpoint
from .data_source import DataSource


class DataSourceRelay:
    """A union of a data source and a stream endpoint to retrieve data points from
    the former and relay them over the latter. This allows the disaggregation of the
    actual datasource from the other processing steps. For example, the relay could
    be deployed with our without an actual processor on another host and the data is
    forwarded over the network to another host running a data source with a
    SimpleRemoteSourceHandler to receive and further process the data. This chain
    could also be continued beyond a single host pair.
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
        """Starts the data source relay along any other objects in this union (data
        source, endpoint). Non-blocking, as the relay is started in the background to
        allow the stopping of it afterward.
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
        """Closes and stops the data source and the endpoint and joins the relay
        thread into the current thread. Can be restarted (and stopped) and arbitrary
        amount of times.
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
        """Actual relay, directly forwards data points from its data source to its
        endpoint (both might be async).
        """
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
    """A union of a data source and a (csv) file handler to retrieve data points form
    the former and storing them in the file. This allows the pre-processing of data
    points from a stream and re-using them at a later time by replaying the file.

    Note that such a relay requires an intact dictionary containing values for all
    fields of the data point header's parameters, i.e. the reduce() function of any
    DataProcessor of the DataSource must not transform it into a numpy vector.
    """

    _logger: logging.Logger

    _data_source: DataSource
    _file: Path
    _header_buffer_size: int
    _headers: tuple[str, ...]
    _separator: str
    _default_missing_value: object

    _relay: threading.Thread
    _started: bool

    def __init__(
        self,
        data_source: DataSource,
        target_file: str,
        name: str = "",
        header_buffer_size: int = 1000,
        overwrite_file: bool = False,
        separator: str = ",",
        default_missing_value: object = "",
    ):
        """Creates a new CSV file relay.

        :param data_source: The data source providing the data points to write to
        file. The processor used by the data source is expected to return data points
        as a dictionary containing all values for all fields in the header's parameter.
        :param target_file: The path to the (new) CSV file. The parent directories
        will be created if not existent.
        :param name: Name of the relay for logging purposes.
        :param header_buffer_size: Number of packets to buffer to generate a common
        header via auto-detection. Note it is not guaranteed that all
        features/headers of all data points in the (possible infinite) stream will be
        discovered, if for example a data point with new features could arrive after
        the discovery is completed.
        :param overwrite_file: Whether the file should be overwritten if it exists.
        :param separator: Separator used in the CSV file.
        :param default_missing_value: Default value if a feature is not present in a
        data point.
        :raises ValueError:
            * If no buffer size is negative or 0.
            * If an invalid CSV separator is provided.
            * If file path provided is not valid.
        """
        self._logger = logging.getLogger(name)
        self._logger.info("Initializing file relay...")

        self._started = False

        if header_buffer_size <= 0:
            raise ValueError("Header buffer size must be greater 0")

        if separator == '"':
            raise ValueError(f"'{separator}' is not allowed as a separator")

        if target_file is None or not target_file:
            raise ValueError("File to write to required.")
        self._file = Path(target_file)
        if self._file.is_dir():
            raise ValueError("File path points to a directory instead of a file.")
        # create parent directories, then touch the file,
        # to check whether it exists and is a valid path
        parent_dir = Path(*self._file.parts[:-1])
        parent_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._file.touch(exist_ok=False)
        except FileNotFoundError:
            raise ValueError("File points to an invalid path.")
        except FileExistsError:
            if not overwrite_file:
                raise ValueError("File already exists and should not be overwritten.")

        self._data_source = data_source
        self._separator = separator
        self._header_buffer_size = header_buffer_size
        self._headers = ()
        self._default_missing_value = default_missing_value

        self._logger.info("File relay initialized.")

    def start(self):
        """Starts the data source relay along the data source itself. Non-blocking,
        as the relay is started in the background to allow the stopping of it afterward.
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
        """Closes and stops the data source and joins the relay thread into the
        current thread. Can be restarted (and stopped) and arbitrary amount of times.

        Note that stopping the relay will not reset the csv file, only recreating it
        will do that.
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
        """Actual relay, directly writes data points from its data source to its
        filehandle in csv style.

        :raises TypeError: Retrieved data point is not of type dictionary. Only
        dictionaries are supported.
        """
        self._logger.info("Starting to relay data points from data source...")
        d_point_counter = 0
        d_point_buffer = []
        do_buffer = True
        header_buffer = OrderedDict()
        self._logger.info("Attempting to discover headers...")
        with open(self._file, "w") as file:
            for d_point in self._data_source:
                try:
                    if not isinstance(d_point, dict):
                        raise TypeError(
                            "Received data point that is not  of type dictionary."
                        )
                    if d_point_counter < self._header_buffer_size:
                        header_buffer.update(OrderedDict.fromkeys(d_point.keys()))
                        d_point_buffer += [d_point]
                    else:
                        if do_buffer:
                            self._headers = tuple(header_buffer)
                            self._logger.info(
                                "Headers found with buffer size of "
                                f"{self._header_buffer_size}: {self._headers}"
                            )
                            file.write(f"{self._separator.join(self._headers)}\n")

                            for d_point_in_buffer in d_point_buffer:
                                values = map(
                                    lambda topic: self._get_value(
                                        d_point_in_buffer, topic
                                    ),
                                    self._headers,
                                )
                                line = self._separator.join(values)
                                file.write(f"{line}\n")
                            do_buffer = False

                        values = map(
                            lambda topic: self._get_value(d_point, topic), self._headers
                        )
                        line = self._separator.join(values)
                        file.write(f"{line}\n")
                    d_point_counter += 1
                except RuntimeError:
                    # stop() was called
                    break
        self._logger.info("Data source exhausted, or relay closing...")

    def _get_value(self, d_point: dict, topic: str) -> str:
        """Retrieves a value from a data point, transforming it into a string to
        prepare it for writing. If it contains csv separators, puts it into quotation
        marks to escape these characters, to make sure it stays a single entry.

        :param d_point: Data point to retrieve value from.
        :param topic: Topic to retrieve from data point.
        :return: Value of data point.
        """
        string_value = str(d_point.get(topic, self._default_missing_value))
        if self._separator in string_value:
            string_value = f'"{string_value}"'
        return string_value

    def __del__(self):
        if self._started:
            self.stop()
