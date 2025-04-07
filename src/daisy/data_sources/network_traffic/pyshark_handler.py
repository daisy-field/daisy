# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Implementations of the data source interface that allows the processing and
provisioning of pyshark packets, either via file inputs, live capture, or a remote
source that generates packets in either fashion.

Author: Jonathan Ackerschewski, Fabian Hofmann
Modified: 04.11.24
"""

import logging
import os
from typing import Iterator, Optional

import pyshark
from natsort import natsorted
from pyshark.capture.capture import TSharkCrashException
from pyshark.capture.file_capture import FileCapture
from pyshark.capture.live_capture import LiveCapture
from pyshark.packet.packet import Packet

from ...data_sources.data_source import DataSource


class LivePysharkDataSource(DataSource):
    """The wrapper implementation to support and handle pyshark live captures as data
    sources. Considered infinite in nature, as it allows the generation of pyshark
    packets, until the capture is stopped.
    Beware that you might have to use root privileges to obtain data from this data
    source. If privileges are missing pyshark might not return any data points or
    warnings.
    """

    _capture: LiveCapture
    _generator: Iterator[Packet]

    def __init__(
        self,
        name: str = "LivePysharkDataSource",
        log_level: int = None,
        interfaces: list = "any",
        bpf_filter: str = "",
    ):
        """Creates a new basic pyshark live capture handler on the given interfaces.

        :param name: Name of data source for logging purposes.
        :param log_level: Logging level of data source.
        :param interfaces: Network interfaces to capture. If not given, runs on all
        interfaces.
        :param bpf_filter: Pcap conform filter to filter or ignore certain traffic.
        """
        super().__init__(name=name, log_level=log_level)

        self._logger.info("Initializing live pyshark data source...")
        self._capture = pyshark.LiveCapture(
            interface=interfaces, bpf_filter=bpf_filter, use_json=True
        )
        self._logger.info("Live pyshark data source initialized.")

    def open(self):
        """Starts the pyshark live caption, initializing the wrapped generator."""
        self._logger.info("Beginning live pyshark capture...")
        self._generator = self._capture.sniff_continuously()

    def close(self):
        """Stops the live caption, essentially disabling the generator. Note that the
        generator might block if one tries to retrieve an object from it after that
        point.
        """
        self._capture.close()
        self._logger.info("Live pyshark capture stopped.")

    def __iter__(self) -> Iterator[Packet]:
        """Returns the wrapped generator. Note this does not catch problems after a
        close() on the handler is called --- one must not retrieve objects after as
        it will result in a deadlock!

        :return: Pyshark generator object for data points as pyshark packets.
        """
        return self._generator


class PcapDataSource(DataSource):
    """The wrapper implementation to support and handle any number of pcap files as
    data sources. Finite: finishes after all files have been processed. Warning: Not
    entirely compliant with the data source abstract class: Neither fully thread
    safe, nor does its __iter__() method shut down after close() has been called. Due
    to its finite nature acceptable however, as this data source is nearly always only
    closed once all data points have been retrieved.
    """

    _pcap_files: list[str]

    _cur_file_counter: int
    _cur_file_handle: Optional[FileCapture]
    _try_counter: int

    def __init__(
        self,
        *file_names: str,
        try_counter: int = 3,
        name: str = "PcapDataSource",
        log_level: int = None,
    ):
        """Creates a new pcap file data source.

        :param file_names: List of paths of single files or directories containing
        .pcap files. Each string should be a name of a file or directory. In case a
        directory is passed, all files ending in .pcap are used. In case a single
        file is passed, it is used regardless of file ending.
        :param try_counter: Number of attempts to open a specific pcap file until
        throwing an exception.
        :param name: Name of data source for logging purposes.
        :param log_level: Logging level of data source.
        """
        super().__init__(name=name, log_level=log_level)

        self._logger.info("Initializing pcap file data source...")
        self._pcap_files = []
        for path in file_names:
            if os.path.isdir(path):
                # Variables in following line are:
                # file_tuple[0] = <sub>-directories; file_tuple[2] = files in directory
                dirs = [(file_tuple[0], file_tuple[2]) for file_tuple in os.walk(path)]
                files = [
                    os.path.join(file_tuple[0], file_name)
                    for file_tuple in dirs
                    for file_name in file_tuple[1]
                    if file_name.endswith(".pcap")
                ]
                if files is None:
                    raise ValueError(
                        f"Directory '{path}' does not contain any .pcap files!"
                    )
                files = natsorted(files)
                self._pcap_files += files
            elif os.path.isfile(path) and path.endswith(".pcap"):
                self._pcap_files.append(path)
        if not self._pcap_files:
            raise ValueError(f"No .pcap files in '{file_names}' could be found.")

        self._cur_file_counter = 0
        self._cur_file_handle = None
        self._try_counter = try_counter
        self._logger.info("Pcap file data source initialized.")

    def open(self):
        """Opens and resets the pcap file data source to the very beginning of the file
        list.
        """
        self._logger.info("Opening pcap file source...")
        self._cur_file_counter = 0
        self._cur_file_handle = None
        self._logger.info("Pcap file source opened.")

    def close(self):
        """Closes any file of the pcap file data source."""
        self._logger.info("Closing pcap file source...")
        if self._cur_file_handle is not None:
            self._cur_file_handle.close()
            self._cur_file_handle = None
        self._logger.info("Pcap file source closed.")

    def _open(self):
        """Opens the next file of the pcap file list, trying to open it several times
        until succeeding (known bug from the pyshark library).
        """
        self._logger.debug("Opening next pcap file...")
        try_counter = 0
        while try_counter < self._try_counter:
            try:
                self._cur_file_handle = pyshark.FileCapture(
                    self._pcap_files[self._cur_file_counter],
                    use_json=True,
                )
                break
            except TSharkCrashException:
                try_counter += 1
                continue
        if try_counter == self._try_counter:
            raise RuntimeError(
                f"Could not open File '{self._pcap_files[self._cur_file_counter]}'"
            )
        self._cur_file_counter += 1
        self._logger.info("Next pcap file opened.")

    def __iter__(self) -> Iterator[Packet]:
        """Returns a generator that yields pyshark packets from each file after
        another, opening and closing them when being actively read.

        :return: Generator object for data points as pyshark packets.
        """
        for _ in self._pcap_files:
            self._open()
            try:
                for packet in self._cur_file_handle:
                    yield packet
            except TSharkCrashException as e:
                logging.warning(
                    f"{e.__class__.__name__}({e}) while reading packets. Closing..."
                )
            self._cur_file_handle.close()
