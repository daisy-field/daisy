# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""A collection of the core interface and base classes for the second component of
any data source (see the docstring of the data source class), that prepares the data
points for further (ML) tasks in streaming-manner. Supports generic functions,
that are chained together in groups of three (following the map, filter,
reduce pattern). Note each kind of may needs its own implementations of the
DataProcessor class.

Author: Fabian Hofmann, Jonathan Ackerschewski
Modified: 16.04.24

TODO Future Work: General purpose data processor for arbitrary processing steps
TODO Future Work: Defining granularity of logging in inits
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class DataProcessor(ABC):
    """An abstract data processor that has to process data points as they come in the
    following three steps:

        - map(): The object is mapped to a dictionary, that includes all the features
        from the data point.

        - filter(): Those features are filtered, based on the need of the
        applications that use the data points in streaming-manner.

        - reduce(): The data point is reduced and converted into a numpy array,
        stripping it of all feature names.

    Any implementation has to funnel all its functionalities through these three
    methods (besides __init__), as they are called through process() by the DataSource.

    Note that any or all of these steps can also be empty, returning the passed
    value; such noops can be useful if processing is distributed across multiple
    processors and machines (e.g. when using it in tandem with DataSourceRelay and a
    SimpleRemoteSourceHandler)
    """

    _logger: logging.Logger

    def __init__(self, name: str = ""):
        """Creates a data processor.

        :param name: Name of processor for logging purposes.
        """
        self._logger = logging.getLogger(name)

    @abstractmethod
    def map(self, o_point: object) -> dict:
        """Deserializes a data object into a dictionary with the data point's feature
        names as keys and values as values.

        :param o_point: Data point as object.
        :return: Data point as dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def filter(self, d_point: dict) -> dict:
        """Filters the data point dictionary by removing features from the vector
        based on a set condition (or filter).

        :param d_point: Data point as dictionary.
        :return: Data point as dictionary.
        """
        raise NotImplementedError

    @abstractmethod
    def reduce(self, d_point: dict) -> np.ndarray:
        """Reduces the data point dictionary into a numpy array/vector, stripped from
        any feature names and redundant information.

        :param d_point: Data point as dictionary.
        :return: Data point as numpy array.
        """
        raise NotImplementedError

    def process(self, o_point: object) -> np.ndarray:
        """Converts and processes a data point object into a feature vector (numpy
        array).

        Method called by DataSource objects as they load and process the data stream.
        Overriding should not be needed as all to be implemented functionality should
        be covered by the three abstract methods that are called by it.

        :param o_point: Data point as object.
        :return: Processed data point as vector.
        """
        return self.reduce(self.filter(self.map(o_point)))


class SimpleDataProcessor(DataProcessor):
    """A simple implementation of the DataProcessor interface. It takes a function
    for each step of map, filter, reduce and calls them in the respective methods.
    """

    _map_fn: Callable[[object], dict]
    _filter_fn: Callable[[dict], dict]
    _reduce_fn: Callable[[dict], np.ndarray]

    def __init__(
        self,
        map_fn: Callable[[object], dict] = lambda o_point: o_point,
        filter_fn: Callable[[dict], dict] = lambda o_point: o_point,
        reduce_fn: Callable[[dict], np.ndarray] = lambda o_point: o_point,
        name: str = "",
    ):
        """Creates a SimpleDataProcessor from the given map/filter/reduce functions. Any
        function not provided will default to noop.

        :param map_fn: The map function, which receives a data point and should map
        it to a dictionary
        :param filter_fn: The filter function, which receives the map output and
        filters/selects its features.
        :param reduce_fn: The reduce function, which receives the filter output and
        transforms into a numpy array.
        :param name: The name for logging purposes.
        """
        super().__init__(name)
        self._map_fn = map_fn
        self._filter_fn = filter_fn
        self._reduce_fn = reduce_fn

    def map(self, o_point: object) -> dict:
        return self._map_fn(o_point)

    def filter(self, d_point: dict) -> dict:
        return self._filter_fn(d_point)

    def reduce(self, d_point: dict) -> np.ndarray:
        return self._reduce_fn(d_point)
