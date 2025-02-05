# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import os
from typing import IO

import pytest

from daisy.data_sources import CSVFileRelay, DataHandler


# noinspection PyTypeChecker
class DataHandlerMock(DataHandler):
    def __init__(self, data_points: list = None):
        super().__init__(data_source=None, data_processor=None)
        self._data_points = data_points if data_points is not None else []

    def open(self):
        pass

    def close(self):
        pass

    def _create_loader(self):
        pass

    def __iter__(self):
        for data_point in self._data_points:
            yield data_point

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


# noinspection PyTypeChecker
def assert_file_content(file: IO, expected_lines: list[str]):
    lines = []
    with open(file, "r") as f:
        for line in f:
            lines.append(line)
    assert len(lines) == len(expected_lines)
    for i in range(len(expected_lines)):
        assert expected_lines[i] == lines[i]


@pytest.fixture(scope="session")
def csv_file_relay_path(tmp_path_factory):
    return tmp_path_factory.mktemp("csv_file_relay")


@pytest.fixture(scope="function")
def csv_file_relay(csv_file_relay_path):
    return CSVFileRelay(
        data_handler=DataHandlerMock(),
        target_file=csv_file_relay_path / "general_csv_file_relay_test_file.csv",
        overwrite_file=True,
    )


@pytest.fixture(scope="function")
def data_points():
    data_points = [
        {"header1": 1, "header2": 2},
        {"header1": 3, "header2": 4},
        {"header3": 5},
        {"header1": 6, "header4": 7},
        {"header5": 8},  # This header is not supposed to be discovered
    ]
    expected_lines = [
        "header1,header2,header3,header4\n",
        "1,2,,\n",
        "3,4,,\n",
        ",,5,\n",
        "6,,,7\n",
        ",,,\n",
    ]
    return data_points, expected_lines


# noinspection PyTypeChecker
class TestCSVFileRelay:
    def test_no_headers_or_discovery_throws_error(self, csv_file_relay_path):
        with pytest.raises(ValueError):
            CSVFileRelay(
                data_handler=DataHandlerMock(),
                target_file=csv_file_relay_path
                / "no_headers_or_discovery_throws_error.csv",
                header_buffer_size=0,
                headers=None,
            )

    def test_param_separator_double_quote_throws_error(self, csv_file_relay_path):
        with pytest.raises(ValueError):
            CSVFileRelay(
                data_handler=DataHandlerMock(),
                target_file=csv_file_relay_path
                / "separator_double_quote_throws_error.csv",
                separator='"',
            )

    def test_param_target_file_none_throws_error(self):
        with pytest.raises(ValueError):
            CSVFileRelay(
                data_handler=DataHandlerMock(),
                target_file=None,
            )

    def test_param_target_file_empty_throws_error(self):
        with pytest.raises(ValueError):
            CSVFileRelay(
                data_handler=DataHandlerMock(),
                target_file="",
            )

    def test_param_target_file_is_dir_throws_error(self, csv_file_relay_path):
        os.mkdir(csv_file_relay_path / "target_file_is_dir_throws_error")
        with pytest.raises(ValueError):
            CSVFileRelay(
                data_handler=DataHandlerMock(),
                target_file=csv_file_relay_path / "target_file_is_dir_throws_error",
            )

    def test_param_overwrite_file_is_false_but_file_exists_throws_error(
        self, csv_file_relay_path
    ):
        (
            csv_file_relay_path
            / "overwrite_file_is_false_but_file_exists_throws_error.csv"
        ).touch()
        with pytest.raises(ValueError):
            CSVFileRelay(
                data_handler=DataHandlerMock(),
                target_file=csv_file_relay_path
                / "overwrite_file_is_false_but_file_exists_throws_error.csv",
            )

    def test_start_two_times_throws_error(self, csv_file_relay):
        try:
            csv_file_relay.start(blocking=False)
            with pytest.raises(RuntimeError):
                csv_file_relay.start(blocking=False)
        finally:
            csv_file_relay.stop()

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

    def test_process_data_point_writes_buffer_to_file(
        self, csv_file_relay_path, data_points
    ):
        csv_file_relay = CSVFileRelay(
            data_handler=DataHandlerMock(),
            target_file=csv_file_relay_path
            / "process_data_point_writes_buffer_to_file.csv",
            header_buffer_size=4,
        )
        file = FileMock(data_points[1])
        for data_point in data_points[0]:
            csv_file_relay._process_data_point(file, data_point)

        assert not csv_file_relay._do_buffer
        file.assert_lines()

    @pytest.mark.parametrize(
        "header_buffer_size,headers,skip_last_datapoint",
        [
            (10, None, True),
            (4, None, False),
            (0, ["header1", "header2", "header3", "header4"], False),
            (10, ["header1", "header2", "header3", "header4"], False),
        ],
        ids=[
            "create_relay_writes_buffer_to_file_if_not_full",
            "param_header_buffer_size_greater_0_discovers_headers",
            "param_headers_are_used_param_hbs_eq_0",
            "param_headers_are_used_param_hbs_gr_0",
        ],
    )
    def test_param_headers_are_used_param_hbs_gr_0(
        self,
        csv_file_relay_path,
        data_points,
        header_buffer_size,
        headers,
        skip_last_datapoint,
    ):
        filepath = csv_file_relay_path / "general_csv_file_relay_test_file.csv"
        if skip_last_datapoint:
            data_points = (data_points[0][:-1], data_points[1][:-1])
        csv_file_relay = CSVFileRelay(
            data_handler=DataHandlerMock(data_points=data_points[0]),
            target_file=filepath,
            header_buffer_size=header_buffer_size,
            headers=headers,
            overwrite_file=True,
        )
        csv_file_relay.start(blocking=True)
        assert_file_content(filepath, data_points[1])
        csv_file_relay.stop()

    def test_param_overwrite_file_is_true_overwrite_file(self, csv_file_relay_path):
        filepath = (
            csv_file_relay_path / "param_overwrite_file_is_true_overwrite_file.csv"
        )
        CSVFileRelay(
            data_handler=DataHandlerMock(),
            target_file=filepath,
        )
        assert os.path.isfile(filepath)
        CSVFileRelay(
            data_handler=DataHandlerMock(),
            target_file=filepath,
            overwrite_file=True,
        )
