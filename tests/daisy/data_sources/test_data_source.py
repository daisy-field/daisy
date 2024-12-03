# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest

from daisy.data_sources import SimpleDataSource, CSVFileDataSource

_csv_filename = "csvfiledatasource_test.csv"


@pytest.fixture(scope="session")
def csv_file(tmp_path_factory, example_dict):
    path = tmp_path_factory.getbasetemp()
    with open(path / _csv_filename, "w") as f:
        f.write(",".join(example_dict.keys()) + "\n")
        f.write(",".join(example_dict.values()) + "\n")
        f.write(",".join(example_dict.values()) + "\n")
    return path / _csv_filename


@pytest.fixture
def example_list():
    return [1, 2, 3, 4, 5]


@pytest.fixture
def simple_data_source(example_list):
    source = SimpleDataSource(generator=example_list.__iter__())
    return source


@pytest.fixture
def csv_data_source(tmp_path_factory, csv_file, request):
    files = request.param
    path = tmp_path_factory.getbasetemp()
    if isinstance(files, str):
        files = str(path / files)
    else:
        files = [str(path / file) for file in files]
    source = CSVFileDataSource(files=files)
    yield source
    source.close()


def test_simple_data_source_iter_produces_iterable(simple_data_source, example_list):
    assert list(simple_data_source.__iter__()) == example_list


class TestCSVFileDataSource:
    @pytest.mark.parametrize(
        "csv_data_source,expected_iterations",
        [
            (_csv_filename, 2),
            ([_csv_filename], 2),
            ([_csv_filename, _csv_filename], 4),
            ("", 2),
        ],
        indirect=["csv_data_source"],
    )
    def test_constructor_files_valid_parameter(
        self, csv_data_source: CSVFileDataSource, example_dict, expected_iterations
    ):
        csv_data_source.open()
        it = csv_data_source.__iter__()
        for _ in range(expected_iterations):
            assert next(it) == example_dict
        with pytest.raises(StopIteration):
            next(it)

    @pytest.mark.parametrize(
        "csv_data_source", ["non-existing_file.csv"], indirect=True
    )
    def test_constructor_files_invalid_parameter(
        self, csv_data_source: CSVFileDataSource
    ):
        csv_data_source.open()
        it = csv_data_source.__iter__()
        with pytest.raises(FileNotFoundError):
            next(it)

    @pytest.mark.parametrize(
        "csv_data_source", ["non-existing_file.csv"], indirect=True
    )
    def test_line_to_dict_valid_input(
        self, csv_data_source: CSVFileDataSource, example_dict
    ):
        assert (
            csv_data_source._line_to_dict(
                list(example_dict.values()), list(example_dict.keys())
            )
            == example_dict
        )

    @pytest.mark.parametrize(
        "csv_data_source", ["non-existing_file.csv"], indirect=True
    )
    def test_line_to_dict_shorter_line(
        self, csv_data_source: CSVFileDataSource, example_dict
    ):
        line = list(example_dict.values())
        line.pop()
        with pytest.raises(ValueError):
            csv_data_source._line_to_dict(line, list(example_dict.keys()))

    @pytest.mark.parametrize(
        "csv_data_source", ["non-existing_file.csv"], indirect=True
    )
    def test_line_to_dict_shorter_header(
        self, csv_data_source: CSVFileDataSource, example_dict
    ):
        header = list(example_dict.keys())
        header.pop()
        with pytest.raises(ValueError):
            csv_data_source._line_to_dict(list(example_dict.values()), header)
