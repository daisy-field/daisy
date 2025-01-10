# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""A number of useful tools that build on top of the data handler module, to provide
relays of data points, either over a network over communication endpoints or directly
to local file(s) on disk. Both wrap around DataHandler and thus process the data
stream as it yields data points. Can be used for arbitrarily large arbitrary data
streams.

Author: Fabian Hofmann, Jonathan Ackerschewski
Modified: 17.04.23
"""

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import IO, Iterable

from daisy.communication import StreamEndpoint
from .data_handler import DataHandler


class DataHandlerRelay:
    """A union of a data handler and a stream endpoint to retrieve data points from
    the former and relay them over the latter. This allows the disaggregation of the
    actual data handler from the other processing steps. For example, the relay could
    be deployed with our without an actual processor on another host and the data is
    forwarded over the network to another host running a data handler with a
    SimpleRemoteDataSource to receive and further process the data. This chain
    could also be continued beyond a single host pair.
    """

    _logger: logging.Logger

    _data_handler: DataHandler
    _endpoint: StreamEndpoint

    _relay: threading.Thread
    _started: bool
    _completed = threading.Event

    def __init__(
        self, data_handler: DataHandler, endpoint: StreamEndpoint, name: str = ""
    ):
        """Creates a new data handler relay.

        :param data_handler: Data handler to relay data points from.
        :param endpoint: Streaming endpoint to which data points are relayed to.
        :param name: Name of data source relay for logging purposes.
        """
        self._logger = logging.getLogger(name)
        self._logger.info("Initializing data handler relay...")

        self._started = False
        self._completed = threading.Event()

        self.data_handler = data_handler
        self._endpoint = endpoint

        self._logger.info("Data handler relay initialized.")

    def start(self, blocking: bool = False):
        """Starts the data handler relay along any other objects in this union (data
        handler, endpoint). Non-blocking, as the relay is started in the background to
        allow the stopping of it afterward.

        :param blocking: Whether the relay should block until all data points have
        been processed.
        :return: Event object to check whether relay has completed processing
        every data point and may be closed. Only useful when calling start()
        non-blocking, otherwise it is implicitly used to wait for completion.
        """
        self._logger.info("Starting data handler relay...")
        if self._started:
            raise RuntimeError("Relay has already been started!")
        self._started = True
        self._completed.clear()
        try:
            self.data_handler.open()
        except RuntimeError:
            pass
        try:
            self._endpoint.start()
        except RuntimeError:
            pass

        self._relay = threading.Thread(target=self._create_relay, daemon=True)
        self._relay.start()
        self._logger.info("Data handler relay started.")

        if blocking:
            self._completed.wait()
            self._logger.info("Relay has processed all data points and may be closed.")
        return self._completed

    def stop(self):
        """Closes and stops the data handler and the endpoint and joins the relay
        thread into the current thread. Can be restarted (and stopped) and arbitrary
        amount of times.
        """
        self._logger.info("Stopping data handler relay...")
        if not self._started:
            raise RuntimeError("Endpoint has not been started!")
        self._started = False
        try:
            self.data_handler.close()
        except RuntimeError:
            pass
        try:
            self._endpoint.stop()
        except RuntimeError:
            pass

        self._relay.join()
        self._logger.info("Data handler relay stopped.")

    def _create_relay(self):
        """Actual relay, directly forwards data points from its data handler to its
        endpoint (both might be async).
        """
        self._logger.info("Starting to relay data points from data handler...")
        for d_point in self._data_handler:
            try:
                self._endpoint.send(d_point)
            except RuntimeError:
                # stop() was called
                break
        self._logger.info("Data source exhausted, or relay closed.")
        self._completed.set()

    def __del__(self):
        if self._started:
            self.stop()


class CSVFileRelay:
    """A union of a data handler and a (csv) file handler to retrieve data points from
    the former and storing them in the file. This allows the pre-processing of data
    points from a stream and re-using them at a later time by replaying the file.

    Note that such a relay requires an intact dictionary containing values for all
    fields of the data point header's parameters.
    """

    _logger: logging.Logger

    _data_handler: DataHandler
    _file: Path
    _header_buffer_size: int
    _headers: tuple[str, ...]
    _headers_provided: bool
    _separator: str
    _default_missing_value: object

    _d_point_counter: int
    _d_point_buffer: list
    _header_buffer: OrderedDict
    _do_buffer: bool

    _relay: threading.Thread
    _started: bool
    _completed = threading.Event

    def __init__(
        self,
        data_handler: DataHandler,
        target_file: str | Path,
        name: str = "",
        header_buffer_size: int = 1000,
        headers: tuple[str, ...] = None,
        overwrite_file: bool = False,
        separator: str = ",",
        default_missing_value: object = "",
    ):
        """Creates a new CSV file relay.

        :param data_handler: The data handler providing the data points to write to
        file. The processor used by the data handler is expected to return data points
        as a dictionary containing all values for all fields in the header's parameter.
        :param target_file: The path to the (new) CSV file. The parent directories
        will be created if not existent.
        :param name: Name of the relay for logging purposes.
        :param header_buffer_size: Number of packets to buffer to generate a common
        header via auto-detection. Note it is not guaranteed that all
        features/headers of all data points in the (possible infinite) stream will be
        discovered, if for example a data point with new features could arrive after
        the discovery is completed. On the other hand, if the buffer size is great
        or equal to the number of points, the entire stream is used for discovery.
        :param headers: If this is provided, the auto header discovery will be turned
        off and the provided headers will be used instead.
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
        self._completed = threading.Event()

        if headers is None and header_buffer_size <= 0:
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

        self._data_handler = data_handler
        self._separator = separator
        if headers is not None:
            self._header_buffer_size = 0
            self._headers = headers
            self._headers_provided = True
        else:
            self._header_buffer_size = header_buffer_size
            self._headers = ()
            self._headers_provided = False
        self._default_missing_value = default_missing_value

        self._d_point_counter = 0
        self._d_point_buffer = []
        self._header_buffer = OrderedDict()
        self._do_buffer = not self._headers_provided

        self._logger.info("File relay initialized.")

    def start(self, blocking: bool = False):
        """Starts the csv file relay along the data source itself. Non-blocking,
        as the relay is started in the background to allow the stopping of it afterward.

        :param blocking: Whether the relay should block until all data points have
        been processed.
        :return: Event object to check whether file relay has completed processing
        every data point and may be closed. Only useful when calling start()
        non-blocking, otherwise it is implicitly used to wait for completion.
        """
        self._logger.info("Starting file relay...")
        if self._started:
            raise RuntimeError("Relay has already been started!")
        self._started = True
        self._completed.clear()
        try:
            self._data_handler.open()
        except RuntimeError:
            pass

        self._relay = threading.Thread(target=self._create_relay, daemon=True)
        self._relay.start()
        self._logger.info("File relay started.")

        if blocking:
            self._completed.wait()
            self._logger.info(
                "File relay has processed all data points and may be closed."
            )
        return self._completed

    def stop(self):
        """Closes and stops the data handler and joins the relay thread into the
        current thread. Can be restarted (and stopped) and arbitrary amount of times.

        Note that stopping the relay will not reset the csv file, only recreating it
        will do that.
        """
        self._logger.info("Stopping file relay...")
        if not self._started:
            raise RuntimeError("Relay has not been started!")
        self._started = False
        try:
            self._data_handler.close()
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
        if self._headers_provided:
            self._logger.info(f"Using headers: {self._headers}")
        else:
            self._logger.info("Attempting to discover headers...")

        with open(self._file, "w") as file:
            if self._headers_provided:
                self._write_line_to_file(file, self._headers)

            try:
                self._iterate_data_points(file=file)
            except RuntimeError:
                # stop() was called
                pass
            if self._d_point_counter <= self._header_buffer_size:
                self._process_buffer(file=file)
        self._logger.info("Data source exhausted, or relay closed.")
        self._completed.set()

    def _iterate_data_points(self, file: IO):
        """Iterates through data points and writes them to the csv file.

        :param file: File to write to
        :raises TypeError: Data point is not of type dictionary. Only dictionaries are supported.
        """
        for d_point in self._data_handler:
            try:
                if not self._started:
                    # stop was called
                    break

                self._process_data_point(file=file, d_point=d_point)
                self._d_point_counter += 1
            except RuntimeError:
                # stop() was called
                break

    def _process_data_point(self, file: IO, d_point: dict):
        """Processes a single data point. Writes it to the buffer or file.

        :param file: File to write to
        :param d_point: data point to process
        :raises TypeError: Data point is not of type dictionary. Only dictionaries are supported.
        """
        if not isinstance(d_point, dict):
            raise TypeError("Received data point that is not of type dictionary.")
        if self._d_point_counter % 100 == 0:
            self._logger.debug(f"Received packet {self._d_point_counter}. ")
        if self._d_point_counter < self._header_buffer_size:
            self._header_buffer.update(OrderedDict.fromkeys(d_point.keys()))
            self._d_point_buffer += [d_point]
        else:
            if self._do_buffer:
                self._process_buffer(file=file)
                self._do_buffer = False
            self._write_data_point_to_file(file, d_point)

    def _process_buffer(self, file: IO):
        """Processes the buffer by detecting the headers and writing its contents to the csv file.

        :param file: File to write to
        """
        self._headers = tuple(self._header_buffer)
        self._logger.info(
            "Headers found with buffer size of "
            f"{self._header_buffer_size}: {self._headers}"
        )
        self._write_line_to_file(file, self._headers)
        for d_point_in_buffer in self._d_point_buffer:
            self._write_data_point_to_file(file, d_point_in_buffer)

    def _write_data_point_to_file(self, file: IO, d_point: dict):
        """Writes a single data point to the csv file.

        :param file: File to write to
        :param d_point: data point to write
        """
        values = map(lambda topic: self._get_value(d_point, topic), self._headers)
        self._write_line_to_file(file, values)

    def _write_line_to_file(self, file: IO, line: Iterable[str]):
        """Writes a single line to the csv file.

        :param file: File to write to
        :param line: Line to write
        """
        file.write(f"{self._separator.join(line)}\n")

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
            string_value = string_value.replace('"', '\\"')
            string_value = f'"{string_value}"'
        return string_value

    def __del__(self):
        if self._started:
            self.stop()
