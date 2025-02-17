# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import typing
from collections import OrderedDict

import pytest

from daisy.data_sources import DataProcessor


@pytest.fixture
def data_processor():
    return DataProcessor()


@pytest.fixture
def flat_dict():
    non_flat_dict = {
        "key1": "value1",
        "key2": {
            "key3": {
                "key4": "value4",
                "key5": ["value5"],
            },
            "key6": "value6",
        },
        "key7": "value7",
        "key8": {
            "key9": "value9",
        },
    }
    flat_dict = {
        "key1": "value1",
        "key2.key3.key4": "value4",
        "key2.key3.key5": ["value5"],
        "key2.key6": "value6",
        "key7": "value7",
        "key8.key9": "value9",
    }
    return non_flat_dict, flat_dict


def test_no_added_func(data_processor, example_dict):
    test_dict = example_dict.copy()
    eval_dict = example_dict.copy()
    assert data_processor.process(o_point=test_dict) == eval_dict


def test_func_order(data_processor, example_dict):
    def concat_number_to_values(number):
        # noinspection PyShadowingNames
        def _concat_number_to_values(data_point, number):
            return {key: data_point[key] + str(number) for key in data_point}

        return lambda x: _concat_number_to_values(x, number)

    test_dict = example_dict.copy()

    for i in range(5):
        data_processor.add_func(func=concat_number_to_values(str(i + 1)))
    result = typing.cast(dict, data_processor.process(o_point=test_dict))

    assert all(map(lambda x: x.endswith("12345"), result.values()))


def test_single_func(data_processor, example_dict):
    test_dict = example_dict.copy()
    data_processor.add_func(func=lambda x: {key: x[key] + "test" for key in x})
    result = typing.cast(dict, data_processor.process(o_point=test_dict))
    assert all(map(lambda x: x.endswith("test"), result.values()))


def test_remove_dict_features(data_processor, example_dict):
    test_dict = example_dict.copy()
    eval_dict = example_dict.copy()
    all_features = list(example_dict.keys())
    to_remove = [all_features[-2], all_features[-5], all_features[-1]]
    data_processor.remove_dict_features(features=to_remove)
    result = typing.cast(dict, data_processor.process(o_point=test_dict))
    for key in to_remove:
        eval_dict.pop(key)
    assert result == eval_dict


def test_keep_dict_feature(data_processor, example_dict):
    test_dict = example_dict.copy()
    all_features = list(example_dict.keys())
    to_keep = [all_features[-2], all_features[-5], all_features[-1]]
    data_processor.keep_dict_feature(features=to_keep)
    result = typing.cast(dict, data_processor.process(o_point=test_dict))
    eval_dict = {key: value for key, value in example_dict.items() if key in to_keep}
    assert result == eval_dict


def test_select_dict_features(data_processor, example_dict):
    test_dict = example_dict.copy()
    all_features = list(example_dict.keys())
    to_select = [all_features[-2], all_features[-5], all_features[-1], "mushroom"]
    data_processor.select_dict_features(features=to_select)
    result = typing.cast(dict, data_processor.process(o_point=test_dict))
    eval_dict = {key: example_dict.get(key, None) for key in to_select}
    assert result == eval_dict


def test_select_dict_features_default_value(data_processor, example_dict):
    test_dict = example_dict.copy()
    all_features = list(example_dict.keys())
    to_select = [all_features[-2], all_features[-5], all_features[-1], "mushroom"]
    data_processor.select_dict_features(features=to_select, default_value="testcase")
    result = typing.cast(dict, data_processor.process(o_point=test_dict))
    eval_dict = {key: example_dict.get(key, "testcase") for key in to_select}
    assert result == eval_dict


def test_flatten_dict(data_processor, flat_dict):
    data_processor.flatten_dict()
    test_dict = flat_dict[0].copy()
    assert data_processor.process(test_dict) == flat_dict[1]


@pytest.mark.parametrize(
    "test_dict",
    [
        OrderedDict(
            [
                ("key1.key2", "value"),
                ("key1", {"key2": "value"}),
            ]
        ),
        OrderedDict(
            [
                ("key1", {"key2": "value"}),
                ("key1.key2", "value"),
            ]
        ),
    ],
)
def test_flatten_dict_value_error(data_processor, test_dict):
    data_processor.flatten_dict()
    with pytest.raises(ValueError):
        data_processor.process(test_dict)
