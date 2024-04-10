# Copyright (C) 2028 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
A collection of interfaces and base classes for data stream generation and preprocessing for further (ML) tasks.
Supports generic generators, but also remote communication endpoints that hand over generic data points in
streaming-manner, and any other implementations of the SourceHandler class. Note each different kind of data needs
its own implementation of the DataProcessor class.

TODO REFVIEW COMENTS @Fabian

Author: Fabian Hofmann, Jonathan Ackerschewski
Modified: 28.07.23

TODO Future Work: Defining granularity of logging in inits
TODO Future Work: Cleanup of inits to eliminate overlap of classes
"""

import logging
from abc import ABC, abstractmethod
from typing import Iterator

from daisy.communication import StreamEndpoint


class SourceHandler(ABC):
    """An abstract wrapper around a generator-like structure that has to yield data points as objects as they come for
    processing. That generator may be infinite or finite, as long as it is bounded on both sides by the following two
    methods that must be implemented:

        - open(): Enables the "generator" to provision data points.

        - close(): Closes the "generator".

    Note that as DataSource, wraps itself a given source handler to retrieve objects, open() and close() do not need to
    be implemented to be idempotent and arbitrarily permutable. Same can be assumed for __iter__() as it will only be
    called when the source handler has been opened already. At the same time, __iter__() must be exhausted after close()
    has been called.
    """

    _logger: logging.Logger

    def __init__(self, name: str = ""):
        """Creates a source handler. Note that this should not enable the immediate generation of data points via
        __iter__() --- this behavior is implemented through open() (see the class documentation for more information).

        :param name: Name of handler for logging purposes.
        """
        self._logger = logging.getLogger(name)

    @abstractmethod
    def open(self):
        """Prepares the handler to be used for data point generation, setting up necessary environment variables,
        starting up background processes to read/generate data, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Closes the handler after which data point generation is no longer available until opened again."""
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[object]:
        """After opened (see open()), returns a generator - either the object itself or creates a new one (e.g. through
        use of the yield statement).

        :return: Generator object for data points as objects.
        """
        raise NotImplementedError


class SimpleSourceHandler(SourceHandler):
    """The simplest productive source handler --- an actual wrapper around a generator that is always open and cannot be
    closed, yielding data points as objects as they are yielded. Can be infinite or finite; no matter, no control over
    the generator is natively supported.
    """

    _generator: Iterator[object]

    def __init__(self, generator: Iterator[object], name: str = ""):
        """Creates a source handler, simply wrapping it around the given generator.

        :param generator: Generator object from which data points are retrieved.
        :param name: Name of handler for logging purposes.
        """
        super().__init__(name)

        self._generator = generator

    def open(self):
        pass

    def close(self):
        pass

    def __iter__(self) -> Iterator[object]:
        """Returns the wrapped generator, requiring neither open() nor close().

        :return: Generator object for data points as objects.
        """
        return self._generator


class SimpleRemoteSourceHandler(SourceHandler):
    """The simple wrapper implementation to support and handle remote streaming endpoints of the Endpoint module as data
    sources. Considered infinite in nature, as it allows the generation of data point objects from a connected
    endpoint, until the client closes the handler.
    """

    _endpoint: StreamEndpoint

    def __init__(self, endpoint: StreamEndpoint, name: str = ""):
        """Creates a new remote source handler from a given stream endpoint. If no endpoint is provided, creates a new
        one instead with basic parameters.

        :param endpoint: Streaming endpoint from which data points are retrieved.
        :param name: Name of handler for logging purposes.
        """
        super().__init__(name)

        self._logger.info("Initializing remote source handler...")
        self._endpoint = endpoint
        self._logger.info("Remote source handler initialized.")

    def open(self):
        """Starts and opens/connects the endpoint of the source handler."""
        self._logger.info("Starting remote data source...")
        try:
            self._endpoint.start()
        except RuntimeError:
            pass
        self._logger.info("Remote data source started.")

    def close(self):
        """Stops and closes the endpoint of the source handler."""
        self._logger.info("Stopping remote data source...")
        try:
            self._endpoint.stop()
        except RuntimeError:
            pass
        self._logger.info("Remote data source stopped.")

    def __iter__(self) -> Iterator[object]:
        """Returns the wrapped endpoint generator, as it supports object retrieval directly.

        :return: Endpoint generator object for data points as objects.
        """
        return self._endpoint.__iter__()
