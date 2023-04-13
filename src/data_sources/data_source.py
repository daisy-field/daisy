"""
    TODO

    Author: Fabian Hofmann
    Modified: 13.04.22
"""

import logging
import queue
import threading
from abc import ABC, abstractmethod
from typing import Iterator, Tuple

from src.communication.message_stream import StreamEndpoint, SINK


# TODO docstrings
# TODO logging
# TODO check order of dict entries, else switch to numpy arrays


class DataSource(ABC):
    """TODO

    """
    _logger: logging.Logger

    _generator: Iterator[object]

    _multithreading: bool
    _thread: threading.Thread
    _buffer: queue.Queue[dict]
    _opened: bool

    def __init__(self, generator: Iterator[object], multithreading: bool = False, buffer_size: int = 1024):
        self._logger = logging.getLogger(f"DataSource")  # FIXME

        self._generator = generator

        self._multithreading = multithreading
        self._buffer = queue.Queue(buffer_size)
        self._opened = False

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

    def open(self):
        """TODO
        """
        if self._opened:
            raise RuntimeError(f"Data source has already been opened!")
        self._opened = True

        if self._multithreading:
            self._thread = threading.Thread(target=self._loader, name="DataSourceLoader", daemon=True)
            self._thread.start()

    def close(self):
        """TODO
        """
        if not self._opened:
            raise RuntimeError(f"Data source has not been opened!")
        self._opened = False

        if self._multithreading:
            self._thread.join()

    def _loader(self):
        """TODO
        """
        for o_point in self._generator:
            while self._opened:
                try:
                    self._buffer.put(self.filter(self.process(o_point)), timeout=10)
                except queue.Full:
                    self._logger.warning(f"{self._thread.name}: Timeout triggered: Buffer full. Retrying...")
            if not self._opened:
                break

    def __iter__(self) -> Iterator[dict]:
        """TODO

        :return:
        """
        if not self._opened:
            raise RuntimeError("Data source has not been opened!")

        if self._multithreading:
            while self._opened:
                try:
                    yield self._buffer.get(timeout=10)
                except queue.Empty:
                    self._logger.warning(f"Timeout triggered: Buffer empty. Retrying...")
        else:
            for o_point in self._generator:
                yield self.filter(self.process(o_point))

    def __del__(self):
        if self._opened:
            self.close()


class RemoteDataSource(DataSource, ABC):
    """TODO

    """
    _endpoint: StreamEndpoint

    def __init__(self, endpoint: StreamEndpoint = None, addr: Tuple[str, int] = ("127.0.0.1", 12000),
                 multithreading: bool = False, buffer_size: int = 1024):
        """TODO

        :param endpoint:
        :param addr:
        :param multithreading:
        :param buffer_size:
        """
        if endpoint is None:
            endpoint = StreamEndpoint(addr=addr, endpoint_type=SINK,
                                      multithreading=multithreading, buffer_size=buffer_size)
        self._endpoint = endpoint

        super().__init__(self._endpoint.__iter__(), multithreading, buffer_size)

    def open(self):
        """TODO
        """
        try:
            self._endpoint.start()
        except RuntimeError:
            pass
        super().open()

    def close(self):
        """TODO
        """
        try:
            self._endpoint.stop()
        except RuntimeError:
            pass
        super().close()

    def __del__(self):
        if self._opened:
            self.close()
