# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""The data processor and relevant generic function intended for it. The data processor
processes individual data points using any number of functions defined by the user.

Author: Fabian Hofmann, Jonathan Ackerschewski
Modified: 18.10.2024
"""
# TODO Future Work: General purpose data processor for arbitrary processing steps
# TODO Future Work: Defining granularity of logging in inits

import logging
from collections.abc import MutableMapping
from typing import Callable


class DataProcessor:
    """A data processor, which is used to process data with various functions. The
    functions can be added to the processor and are carried out one after the other
    in the provided order.
    """

    _logger: logging.Logger
    _functions: list[Callable]

    def __init__(self, name: str = ""):
        """Creates a data processor.

        :param name: Name of processor for logging purposes.
        """
        self._logger = logging.getLogger(name)
        self._functions = []

    def add_func(self, func: Callable[[object], object]):
        """Adds a function to the processor.

        :param func: The function to add to the processor.
        """
        self._functions.append(func)
        return self

    def process(self, o_point: object) -> object:
        """Processes the given data point using the provided functions. The functions
        are carried out in the order they were added. If no functions were provided,
        the data point is returned unchanged.
        """
        point = o_point
        for func in self._functions:
            point = func(point)
        return point


def remove_feature(d_point: dict, f_features: list) -> dict:
    """Takes a data point as a dictionary and removes all given features from it.

    :param d_point: Dictionary of data point.
    :param f_features: List of features to remove.
    :return: Dictionary of data point with features removed.
    """
    for feature in f_features:
        d_point.pop(feature, None)
    return d_point


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
