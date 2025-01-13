# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import os
from typing import IO

import pytest

from daisy.data_sources import CSVFileRelay, DataHandler


# CSVFileRelay tests
# header_buffer_size = 0
# header_buffer_size > 0
# headers
# overwrite test
# separator test
# default_missing_value test
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


class FileMock(IO):
    _expected_lines: list[str]
    _written_lines: list[str]

    def __init__(self, expected_lines: list[str]):
        self._expected_lines = expected_lines
        self._written_lines = []

    def assert_lines(self):
        assert len(self._expected_lines) == len(self._written_lines)
        for i in range(len(self._expected_lines)):
            assert self._written_lines[i] == self._expected_lines[i]

    def close(self):
        pass

    def fileno(self):
        pass

    def flush(self):
        pass

    def isatty(self):
        pass

    def read(self, __n=-1):
        pass

    def readable(self):
        pass

    def readline(self, __limit=-1):
        pass

    def readlines(self, __hint=-1):
        pass

    def seek(self, __offset, __whence=0):
        pass

    def seekable(self):
        pass

    def tell(self):
        pass

    def truncate(self, __size=None):
        pass

    def writable(self):
        pass

    def write(self, __s):
        lines = __s.splitlines(keepends=True)
        self._written_lines.extend(lines)

    def writelines(self, __lines):
        for line in __lines:
            self.write(line)

    def __next__(self):
        pass

    def __iter__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, __type, __value, __traceback):
        pass


@pytest.fixture(scope="function")
def csv_file_relay(tmp_path_factory):
    path = tmp_path_factory.getbasetemp()
    return CSVFileRelay(
        data_handler=DataHandlerMock(),
        target_file=path / "general_csv_file_relay_test_file.csv",
        overwrite_file=True,
    )


class TestCSVFileRelay:
    def test_no_headers_or_discovery_throws_error(self, tmp_path_factory):
        path = tmp_path_factory.getbasetemp()
        with pytest.raises(ValueError):
            CSVFileRelay(
                data_handler=DataHandlerMock(),
                target_file=path / "no_headers_or_discovery_throws_error.csv",
                header_buffer_size=0,
                headers=None,
            )

    def test_separator_double_quote_throws_error(self, tmp_path_factory):
        path = tmp_path_factory.getbasetemp()
        with pytest.raises(ValueError):
            CSVFileRelay(
                data_handler=DataHandlerMock(),
                target_file=path / "separator_double_quote_throws_error.csv",
                separator='"',
            )

    def test_target_file_none_throws_error(self):
        with pytest.raises(ValueError):
            CSVFileRelay(
                data_handler=DataHandlerMock(),
                target_file=None,
            )

    def test_target_file_empty_throws_error(self):
        with pytest.raises(ValueError):
            CSVFileRelay(
                data_handler=DataHandlerMock(),
                target_file="",
            )

    def test_target_file_is_dir_throws_error(self, tmp_path_factory):
        path = tmp_path_factory.getbasetemp()
        os.mkdir(path / "target_file_is_dir_throws_error")
        with pytest.raises(ValueError):
            CSVFileRelay(
                data_handler=DataHandlerMock(),
                target_file=path / "target_file_is_dir_throws_error",
            )

    def test_target_file_invalid_path_throws_error(
        self, tmp_path_factory
    ):  # TODO test if this works on Linux
        path = tmp_path_factory.getbasetemp()
        with pytest.raises(ValueError):
            CSVFileRelay(
                data_handler=DataHandlerMock(),
                target_file=path
                / "target_file_invalid_path_throws_error"
                / ("a" * 4096 + ".csv"),
            )

    def test_overwrite_file_is_false_but_file_exists_throws_error(
        self, tmp_path_factory
    ):
        path = tmp_path_factory.getbasetemp()
        (path / "overwrite_file_is_false_but_file_exists_throws_error.csv").touch()
        with pytest.raises(ValueError):
            CSVFileRelay(
                data_handler=DataHandlerMock(),
                target_file=path
                / "overwrite_file_is_false_but_file_exists_throws_error.csv",
            )

    def test_start_two_times_throws_error(self, csv_file_relay):
        csv_file_relay.start(blocking=False)
        with pytest.raises(RuntimeError):
            csv_file_relay.start(blocking=False)

    def test_stop_without_start_throws_error(self, csv_file_relay):
        with pytest.raises(RuntimeError):
            csv_file_relay.stop()

    def test_write_line_to_file_writes_line(self, csv_file_relay):
        file = FileMock(expected_lines=["This,is,a,test,line\n"])
        csv_file_relay._write_line_to_file(file, ["This", "is", "a", "test", "line"])
        file.assert_lines()

    def test_process_data_point_writes_to_buffer(self, csv_file_relay):
        file = FileMock([])
        data_points = [{"data_point": 1}]
        for data_point in data_points:
            csv_file_relay._process_data_point(file, data_point)

        assert csv_file_relay._d_point_buffer == data_points
        assert csv_file_relay._do_buffer

    def test_process_data_point_writes_buffer_to_file(self, tmp_path_factory):
        path = tmp_path_factory.getbasetemp()
        csv_file_relay = CSVFileRelay(
            data_handler=DataHandlerMock(),
            target_file=path / "process_data_point_writes_buffer_to_file.csv",
            header_buffer_size=2,
        )
        file = FileMock(
            expected_lines=[
                "data_point,header1,header2\n",
                "1,value1,\n",
                "2,,value2\n",
                "3,,\n",
            ]
        )
        data_points = [
            {"data_point": 1, "header1": "value1"},
            {"data_point": 2, "header2": "value2"},
            {"data_point": 3, "header3": "value3"},
        ]
        for data_point in data_points:
            csv_file_relay._process_data_point(file, data_point)

        assert not csv_file_relay._do_buffer
        file.assert_lines()
