# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Base class data processor and its relevant pre-built processing function steps for
generic data in the form of objects and dictionaries. The data processor processes
individual data points using any number of user-defined functions. These functions can
either be defined from scratch or chosen from a list of pre-built ones.

Author: Fabian Hofmann, Jonathan Ackerschewski
Modified: 04.11.2024
"""

import json
import logging
from collections.abc import MutableMapping
from typing import Callable, Self
from typing_extensions import deprecated

import numpy as np


class DataProcessor:
    """Base class for generic data stream processing, in pipelined fashion. The
    processing steps are implemented as functions independent of each other, carried
    out in one specific order for each data point in the stream. For these functions,
    there is also a list of pre-built processing steps as methods for ease of use,
    but any customized function can also be added using add_func().

    Extension of this base class merely manipulate _functions through the use of
    add_func() to provide additional pre-built processing steps.
    """

    _logger: logging.Logger
    _functions: list[Callable]

    def __init__(self, name: str = ""):
        """Creates a data processor.

        :param name: Name of processor for logging purposes.
        """
        self._logger = logging.getLogger(name)
        self._functions = []

    def add_func(self, func: Callable[[object], object]) -> Self:
        """Adds a function to the processor to the end of its function list.

        :param func: The function to add to the processor.
        """
        self._functions.append(func)
        return self

    def remove_dict_features(self, features: list) -> Self:
        """Adds a function to the processor that takes a data point as a dictionary and
        removes all given features from it.

        :param features: List of features to remove.
        """

        def remove_features_func(d_point: dict) -> dict:
            for feature in features:
                d_point.pop(feature, None)
            return d_point

        return self.add_func(lambda d_point: remove_features_func(d_point))

    def keep_dict_feature(self, features: list) -> Self:
        """Adds a function to the processor that takes a data point as a dictionary and
        keeps all given features.

        :param features: List of features to keep.
        :return: Dictionary of data point with features kept.
        """
        return self.add_func(
            lambda d_point: {
                key: value for key, value in d_point.items() if key in features
            }
        )

    def select_dict_features(self, features: list, default_value=None) -> Self:
        """Adds a function to the processor that takes a data point which is a
        dictionary and selects features to keep. If a feature should be kept but isn't
        present in the data point, it will be added with the default value.

        :param features: List of features to select.
        :param default_value: Default value if feature is not in data point.
        """
        return self.add_func(
            lambda d_point: {
                feature: d_point.get(feature, default_value) for feature in features
            }
        )

    def flatten_dict(self, separator: str = ".") -> Self:
        """Adds a function to the processor that creates a flat dictionary
        (a dictionary without sub-dictionaries) from the given dictionary. The keys
        of sub-dictionaries are merged into the parent dictionary by combining the
        keys and adding a separator:
        {a: {b: c, d: e}, f: g} becomes {a.b: c, a.d: e, f: g} assuming the separator
        as '.'. However, redundant parent keys are greedily eliminated from the
        dictionary and further collisions cause an error.

        :param separator: Separator to use.
        """

        def flatten_dict_func(
            dictionary: (dict, list),
            par_key: str = "",
        ) -> dict:
            """See the parent function for more information on its behavior.

            :param dictionary: Dictionary to flatten.
            :param par_key: Key of the parent dictionary.
            :raises ValueError: If there are key-collisions by greedily flattening the
            dictionary.
            """
            items = {}
            for key, val in dictionary.items():
                cur_key = par_key + separator + key if par_key != "" else key
                if isinstance(val, MutableMapping):
                    sub_items = flatten_dict_func(val, par_key=cur_key)
                    for subkey in sub_items.keys():
                        if subkey in items:
                            raise ValueError(
                                f"Key collision in dictionary "
                                f"({subkey, sub_items[subkey]} "
                                f"vs {subkey, items[subkey]})!"
                            )
                    items.update(sub_items)
                else:
                    if cur_key in items:
                        raise ValueError(
                            f"Key collision in dictionary ({cur_key, val} "
                            f"vs {cur_key, items[cur_key]})!"
                        )
                    items.update({cur_key: val})
            return items

        return self.add_func(lambda d_point: flatten_dict_func(d_point))

    def dict_to_array(self, nn_aggregator: Callable[[str, object], object]) -> Self:
        """Adds a function to the processor that takes a data point which is a
        dictionary and lazily transforms it into a numpy array without further
        processing, aggregating any value that is list into a singular value based on a
        pre-defined function that operates only on each singular feature.

        :param nn_aggregator: Aggregator, which maps non-numerical features to integers
        or floats on a value-by-value basis.
        """

        # noinspection DuplicatedCode
        def dict_to_array_func(d_point: dict) -> np.ndarray:
            l_point = []
            for key, value in d_point.items():
                if not isinstance(value, int | float):
                    value = nn_aggregator(key, value)
                try:
                    if np.isnan(value):
                        value = 0
                except TypeError as e:
                    raise ValueError(f"Invalid k/v pair: {key}, {value}") from e
                l_point.append(value)
            return np.asarray(l_point)

        return self.add_func(lambda d_point: dict_to_array_func(d_point))

    def dict_to_json(self):
        """Adds a function to the processor that takes a data point which is a
        dictionary and converts it to a JSON string.
        """
        return self.add_func(lambda d_point: json.dumps(d_point))

    def process(self, o_point: object) -> object:
        """Processes the given data point using the provided functions. The functions
        are carried out in the order they were added. If no functions were provided,
        the data point is returned unchanged (noop).

        Note process() is usually called by the data handler during data processing and
        should not be called directly.
        """
        point = o_point
        for func in self._functions:
            point = func(point)
        return point


@deprecated("Use DataProcessor.remove_features() instead")
def remove_feature(d_point: dict, f_features: list) -> dict:
    """Takes a data point as a dictionary and removes all given features from it.

    :param d_point: Dictionary of data point.
    :param f_features: List of features to remove.
    :return: Dictionary of data point with features removed.
    """
    for feature in f_features:
        d_point.pop(feature, None)
    return d_point


@deprecated("Use DataProcessor.keep_features() instead")
def keep_feature(d_point: dict, f_features: list) -> dict:
    """Takes a data point as a dictionary and removes all features not in the given
    list.

    :param d_point: Dictionary of data point.
    :param f_features: List of features to keep.
    :return: Dictionary of data point with features kept.
    """
    return {key: value for key, value in d_point.items() if key in f_features}


@deprecated("Use DataProcessor.select_features() instead")
def select_feature(d_point: dict, f_features: list, default_value=None) -> dict:
    """Takes a data point as a dictionary and selects features to keep. If a feature
    should be kept but isn't present in the data point, it will be added with the
    default value.

    :param d_point: Dictionary of data point.
    :param f_features: List of features to select.
    :param default_value: Default value if feature is not in original data point.
    :return: Dictionary of data point with selected features.
    """
    return {feature: d_point.get(feature, default_value) for feature in f_features}


@deprecated("Use DataProcessor.flatten() instead")
def flatten_dict(
    dictionary: (dict, list),
    separator: str = ".",
    par_key: str = "",
) -> dict:
    """Creates a flat dictionary (a dictionary without sub-dictionaries) from the
    given dictionary. The keys of sub-dictionaries are merged into the parent
    dictionary by combining the keys and adding a separator: {a: {b: c, d: e}, f: g}
    becomes {a.b: c, a.d: e, f: g} assuming the separator as '.'. However,
    redundant parent keys are greedily eliminated from the dictionary.

    :param dictionary: Dictionary to flatten.
    :param separator: Separator to use.
    :param par_key: Key of the parent dictionary.
    :return: Flat dictionary with keys merged and seperated using the separator.
    :raises ValueError: If there are key-collisions by greedily flattening the
    dictionary.
    """
    items = {}
    for key, val in dictionary.items():
        cur_key = (
            par_key + separator + key
            if par_key != "" and not key.startswith(par_key + separator)
            else key
        )
        if isinstance(val, MutableMapping):
            # noinspection PyDeprecation
            sub_items = flatten_dict(val, par_key=cur_key, separator=separator)
            for subkey in sub_items.keys():
                if subkey in items:
                    raise ValueError(
                        f"Key collision in dictionary "
                        f"({subkey, sub_items[subkey]} vs {subkey, items[subkey]})!"
                    )
            items.update(sub_items)
        else:
            if cur_key in items:
                raise ValueError(
                    f"Key collision in dictionary ({cur_key, val} "
                    f"vs {cur_key, items[cur_key]})!"
                )
            items.update({cur_key: val})
    return items
