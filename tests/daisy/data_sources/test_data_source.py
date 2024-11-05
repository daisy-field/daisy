# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import logging
import unittest
from daisy.data_sources import SimpleDataSource, CSVFileDataSource


class TestSimpleDataSource(unittest.TestCase):
    def setUp(self):
        self._testlist = [1, 2, 3, 4, 5]
        self._source = SimpleDataSource(self._testlist.__iter__())

    def tearDown(self):
        self._source = None

    def test_iter_produces_iterable(self):
        self.assertListEqual(list(self._source.__iter__()), self._testlist)


class TestSimpleRemoteDataSource(unittest.TestCase):
    pass


class TestCSVFileDataSourceConstructor(unittest.TestCase):
    def setUp(self):
        self._result_dict = {
            "header1": "data1",
            "header2": "data2",
            "header3": "data3",
            "header4": "data4",
            "header5": "data5",
            "header6": "data6",
            "header7": "data7",
            "header8": "data8",
        }

    def tearDown(self):
        self._source.close()

    def test_constructor_string_parameter(self):
        self._source = CSVFileDataSource(
            files="tests/resources/csvfiledatasource_test_file.csv"
        )
        self._source.open()
        it = self._source.__iter__()
        for i in range(2):
            with self.subTest(i=i):
                self.assertDictEqual(next(it), self._result_dict)
        self.assertRaises(StopIteration, next, it)

    def test_constructor_list_parameter_single_file(self):
        self._source = CSVFileDataSource(
            files=["tests/resources/csvfiledatasource_test_file.csv"]
        )
        self._source.open()
        it = self._source.__iter__()
        for i in range(2):
            with self.subTest(i=i):
                self.assertDictEqual(next(it), self._result_dict)
        self.assertRaises(StopIteration, next, it)

    def test_constructor_list_parameter_multiple_files(self):
        self._source = CSVFileDataSource(
            files=[
                "tests/resources/csvfiledatasource_test_file.csv",
                "tests/resources/csvfiledatasource_test_file.csv",
            ]
        )
        self._source.open()
        it = self._source.__iter__()
        for i in range(4):
            with self.subTest(i=i):
                self.assertDictEqual(next(it), self._result_dict)
        self.assertRaises(StopIteration, next, it)

    def test_non_existing_file(self):
        self._source = CSVFileDataSource(files="tests/resources/non-existing.csv")
        self._source.open()
        self.assertRaises(FileNotFoundError, lambda: list(self._source.__iter__()))


class TestCSVFileDataSource(unittest.TestCase):
    def setUp(self):
        self._result_dict = {
            "header1": "data1",
            "header2": "data2",
            "header3": "data3",
            "header4": "data4",
            "header5": "data5",
            "header6": "data6",
            "header7": "data7",
            "header8": "data8",
        }
        self._source = CSVFileDataSource(files="")

    def tearDown(self):
        self._source = None

    def test_line_to_dict_valid_input(self):
        line = ["data1", "data2", "data3", "data4", "data5", "data6", "data7", "data8"]
        header = [
            "header1",
            "header2",
            "header3",
            "header4",
            "header5",
            "header6",
            "header7",
            "header8",
        ]
        self.assertDictEqual(
            self._source._line_to_dict(line, header), self._result_dict
        )

    def test_line_to_dict_shorter_line(self):
        line = ["data1", "data2", "data3", "data4", "data5", "data6"]
        header = [
            "header1",
            "header2",
            "header3",
            "header4",
            "header5",
            "header6",
            "header7",
            "header8",
        ]
        with self.assertLogs(self._source._logger, logging.WARN):
            self.assertRaises(IndexError, self._source._line_to_dict, line, header)

    def test_line_to_dict_shorter_header(
        self,
    ):  # TODO is this supposed to be the expected behaviour?
        line = ["data1", "data2", "data3", "data4", "data5", "data6", "data7", "data8"]
        header = ["header1", "header2", "header3", "header4", "header5", "header6"]
        self._result_dict.pop("header7")
        self._result_dict.pop("header8")
        with self.assertLogs(self._source._logger, logging.WARN):
            self.assertDictEqual(
                self._source._line_to_dict(line, header), self._result_dict
            )


if __name__ == "__main__":
    unittest.main()
