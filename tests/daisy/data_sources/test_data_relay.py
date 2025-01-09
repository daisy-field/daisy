# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest

from daisy.data_sources import CSVFileRelay, DataHandler


# CSVFileRelay tests
# header = None && header_buffer_siue <= 0 -> ValueError
# separator = " -> ValueError
# target_file = None || target_file = "" -> ValueError
# Path points to dir -> ValueError
# Invalid Path -> ValueError
# File exists + overwrite_file = False -> ValueError
# header_buffer_size = 0
# header_buffer_size > 0
# headers
# overwrite test
# separator test
# default_missing_value test
# start() 2x -> RuntimeError
# stop not started -> RuntimeError
# write line to file actually writes a line using the separator and \n
# get_value correctly converts the feature to string (multiple tests!!)
# process_data_point wrong data type -> TypeError
# process_data_point adds to buffer if still discovering
# process_data_point writes buffer, if buffer full
# create_relay -> if done or stopped before buffer is full-> Write buffer


class DataHandlerMock(DataHandler):
    def __init__(self):
        super().__init__(data_source=None, data_processor=None)

    def open(self):
        pass

    def close(self):
        pass

    def _create_loader(self):
        pass

    def __iter__(self):
        # TODO yield something
        pass

    def __del__(self):
        pass


class TestCSVFileRelay:
    def test_no_headers_or_discovery_throws_error(self, tmp_path_factory):
        with pytest.raises(ValueError):
            path = tmp_path_factory.getbasetemp()
            CSVFileRelay(
                data_handler=DataHandlerMock(),
                target_file=path / "no_headers_or_discovery_throws_error.csv",
                header_buffer_size=0,
                headers=None,
            )
