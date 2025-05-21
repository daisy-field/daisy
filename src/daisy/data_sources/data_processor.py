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

import numpy as np
from typing_extensions import deprecated

from .events import EventHandler


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

    def __init__(self, name: str = "DataProcessor", log_level: int = None):
        """Creates a data processor.

        :param name: Name of processor for logging purposes.
        :param log_level: Logging level of processor.
        """
        self._logger = logging.getLogger(name)
        if log_level:
            self._logger.setLevel(log_level)
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
                cur_key = (
                    par_key + separator + key
                    if par_key != "" and not key.startswith(par_key + separator)
                    else key
                )
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

    def add_event_handler(self, event_handler: EventHandler) -> Self:
        """Adds the provided event handler to the processor to evaluate the data points
        based on the event handlers events.

        :param event_handler: Event handler to use.
        """
        return self.add_func(lambda d_point: event_handler.evaluate(d_point))

    def enrich_dict_feature(
        self,
        feature: str,
        enricher: Callable[[dict], object],
        force_overwrite: bool = False,
        suppress_errors: bool = False,
    ) -> Self:
        """Enriches the data point by adding the given feature using the provided
        enricher function. The enricher function is fed the data point and should
        return the desired value for the feature. If the feature already exists, an
        error will be raised. If suppress errors is enabled, an info log will be
        created instead. All errors and logs can be suppressed using force overwrite.

        :param feature: Feature to enrich.
        :param enricher: Enricher function to be applied to the feature.
        :param force_overwrite: Whether to overwrite an existing feature.
        :param suppress_errors: Whether to suppress errors during enrichment
        :raises KeyError: If the feature already exists and is not suppressed.
        """

        def enrich_dict_feature_func(d_point: dict) -> dict:
            if feature in d_point and not force_overwrite:
                if not suppress_errors:
                    raise KeyError(f"Feature '{feature}' already exists")
                self._logger.info(f"Feature '{feature}' already in data point.")
            d_point[feature] = enricher(d_point)
            return d_point

        return self.add_func(lambda d_point: enrich_dict_feature_func(d_point))

    def merge_dict(
        self,
        new_features: dict,
        force_overwrite: bool = False,
        suppress_errors: bool = False,
    ) -> Self:
        """Merges the provided dictionary into the data point. If the feature already
        exists, an error will be raised. If suppress errors is enabled, an info log
        will be created instead. All errors and logs can be suppressed using force
        overwrite.

        :param new_features: Dictionary to merge into the data point.
        :param force_overwrite: Whether to overwrite an existing feature.
        :param suppress_errors: Whether to suppress errors during enrichment.
        :raises KeyError: If the feature already exists and is not suppressed.
        """

        def merge_dict_func(d_point: dict) -> dict:
            for key, value in new_features.items():
                if key in d_point and not force_overwrite:
                    if not suppress_errors:
                        raise KeyError(f"Feature '{key}' already exists")
                    self._logger.info(f"Feature '{key}' already in data point.")
                d_point[key] = value
            return d_point

        return self.add_func(lambda d_point: merge_dict_func(d_point))

    def cast_dict_features(
        self,
        features: str | list[str],
        cast: Callable[[object], object] | list[Callable[[object], object]],
    ) -> Self:
        """Casts the given feature(s) using the provided cast function(s). For both
        features and casts a single or multiple values can be provided. If only one
        cast function is provided with multiple features, all given features will
        be cast using the single cast function. Otherwise, the same amount of cast
        functions and features should be provided.


        :param features: The features to cast.
        :param cast: The cast function(s) to apply.
        :raises ValueError: If the amount of features and cast functions is not equal.
        (Only if more than one cast function is provided.)
        """

        def cast_func(d_point: dict) -> dict:
            if isinstance(features, list) and isinstance(cast, list):
                if len(features) != len(cast):
                    raise ValueError(
                        f"Features and casts must have the same length. "
                        f"Features has {len(features)} and casts has {len(cast)} items."
                    )
            _features = features
            if isinstance(features, str):
                _features = [features]

            for i in range(len(_features)):
                if isinstance(cast, list):
                    d_point[_features[i]] = cast[i](d_point[_features[i]])
                else:
                    d_point[_features[i]] = cast(d_point[_features[i]])
            return d_point

        return self.add_func(lambda d_point: cast_func(d_point))

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

    def shrink_payload_add_labels(
        self,
    ) -> Self:
        """ """

        def shrink_payload_func(d_point: dict) -> dict:
            label = d_point["label"]
            del d_point["label"]

            if d_point["ip.dst"]=="141.23.65.122" or d_point["ip.src"]=="141.23.65.122":
                d_point["malicious_ip"] = 1
            else:
                d_point["malicious_ip"] = 0
            del d_point["ip.dst"]
            del d_point["ip.src"]
            # if "udp.payload" in d_point and "tcp.payload" in d_point:
            #     d_point["payload_length"] = len(d_point["udp.payload"]) + len(
            #         d_point["udp.payload"]
            #     )
            #     del d_point["udp.payload"]
            #     del d_point["tcp.payload"]
            # else:
            #     raise KeyError("Feature udp.payload or tcp.payload does not exists")
            if label == "attack-download":
                d_point["label"] = True
            else:
                d_point["label"] = False

            logging.warning(d_point)
            return d_point

        return self.add_func(lambda d_point: shrink_payload_func(d_point))


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
