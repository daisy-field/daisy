"""
    TODO

    Author: Fabian Hofmann
    Modified: 13.04.22
"""

import queue
import threading
from abc import ABC, abstractmethod
from typing import Iterator


# TODO refactor via abstract classes
# TODO further cleanup, docstrings, typehints


class DataSource(ABC):
    """TODO

    """
    _generator: Iterator[object]

    _multithreading: bool
    _thread: threading.Thread
    _buffer: queue.Queue

    def __init__(self, generator: Iterator[object], multithreading: bool = False, buffer_size: int = 1024):
        self._generator = generator

        self._multithreading = multithreading
        if multithreading:
            self._buffer = queue.Queue(buffer_size)
            self._thread = threading.Thread(target=self._loader, name="DataSourceLoader", daemon=True)
            self._thread.start()

    @abstractmethod
    def process(self, o_point: object) -> dict:
        """TODO

        :param o_point:
        """
        raise NotImplementedError

    @abstractmethod
    def filter(self, d_point: dict) -> dict:
        """TODO

        :param d_point:
        """
        raise NotImplementedError

    def _loader(self):
        """TODO
        """
        for o_point in self._generator:
            self._buffer.put(self.filter(self.process(o_point)))

    def __iter__(self) -> Iterator[dict]:
        """TODO

        :return:
        """
        if self._multithreading:
            yield self._buffer.get()
        else:
            for o_point in self._generator:
                yield self.filter(self.process(o_point))
