# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Base class data processor and its relevant processing function steps for generic
data in the form of objects and dictionaries. The data processor processes individual
data points using any number of functions defined by the user.

Author: Fabian Hofmann, Jonathan Ackerschewski
Modified: 18.10.2024
"""
# TODO Future Work: Defining granularity of logging in inits

import logging
from collections.abc import MutableMapping
from typing import Callable, Self
from warnings import deprecated


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

    def remove_features(self, features: list) -> Self:
        """Adds a function to the processor that takes a data point as a dictionary and
        removes all given features from it.

        :param features: List of features to remove.
        """

        def rem_feats_func(d_point: dict) -> dict:
            for feature in features:
                d_point.pop(feature, None)
            return d_point

        return self.add_func(lambda d_point: rem_feats_func(d_point))

    def keep_feature(self, features: list) -> Self:
        """Takes a data point as a dictionary and removes all features not in the given
        list.

        :param features: List of features to keep.
        :return: Dictionary of data point with features kept.
        """
        return self.add_func(
            lambda d_point: {
                key: value for key, value in d_point.items() if key in features
            }
        )

    def select_features(self, features: list, default_value=None) -> Self:
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


def flatten_dict(
    dictionary: (dict, list),
    seperator: str = ".",
    par_key: str = "",
) -> dict:
    """Creates a flat dictionary (a dictionary without sub-dictionaries) from the
    given dictionary. The keys of sub-dictionaries are merged into the parent
    dictionary by combining the keys and adding a seperator: {a: {b: c, d: e}, f: g}
    becomes {a.b: c, a.d: e, f: g} assuming the seperator as '.'. However,
    redundant parent keys are greedily eliminated from the dictionary.

    :param dictionary: The dictionary to flatten.
    :param seperator: The seperator to use.
    :param par_key: The key of the parent dictionary.
    :return: A flat dictionary with keys merged and seperated using the seperator.
    :raises ValueError: If there are key-collisions by greedily flattening the
    dictionary.
    """
    items = {}
    for key, val in dictionary.items():
        cur_key = (
            par_key + seperator + key
            if par_key != "" and not key.startswith(par_key + seperator)
            else key
        )
        if isinstance(val, MutableMapping):
            sub_items = flatten_dict(val, par_key=cur_key, seperator=seperator)
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
