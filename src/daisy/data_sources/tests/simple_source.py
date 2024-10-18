# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Collection of various function to test simple data point handling of the data handler
class. These test-functions can be called directly, with the main on the bottom
adjusted for each test case.

Author: Fabian Hofmann
Modified: 06.06.24
"""

import logging

from daisy.data_sources import (
    DataHandler,
    CSVFileRelay,
    DataProcessor,
    SimpleDataSource,
)


def source_writer():
    """Creates and starts a file relay to write the contents of an entire data handler
    to a csv file that runs until processing all data points.
    """
    _list = [{"a": 2}, {"a": 3}, {"a": 3}, {"a": 4}, {"a": 5}]
    source = SimpleDataSource(iter(_list))
    processor = DataProcessor()

    with DataHandler(
        data_source=source, data_processor=processor, multithreading=True
    ) as handler:
        relay = CSVFileRelay(
            data_handler=handler,
            target_file="test_file.csv",
            overwrite_file=True,
        )
        relay.start(blocking=True)


def source_status():
    """Creates and starts a finite data handler to continuously yield objects,
    while checking the status of the data handler, that must be unset until the loop
    is complete."""
    _list = [{"a": 2}, {"a": 3}, {"a": 3}, {"a": 4}, {"a": 5}]
    source = SimpleDataSource(iter(_list))
    processor = DataProcessor()

    d = DataHandler(data_source=source, data_processor=processor, multithreading=True)
    status = d.open()
    for _ in d:
        print(status)
    print(status)
    d.close()


def list_source():
    """Creates and starts a data handler from a finite list of elements to be yielded,
    wrapped in a simple data source and using noops as processing steps.
    """
    _list = [{"a": 2}, {"a": 3}, {"a": 3}, {"a": 4}, {"a": 5}]
    source = SimpleDataSource(iter(_list))
    processor = DataProcessor()

    with DataHandler(
        data_source=source, data_processor=processor, multithreading=True
    ) as handler:
        for item in handler:
            print(item)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    # source_status()
    # list_source()
    source_writer()
