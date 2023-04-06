import ctypes
import logging
import pickle
import queue
import socket
import threading
from enum import Enum
from typing import Tuple


class StreamEndpoint:
    class EndpointType(Enum):
        SOURCE = 0
        SINK = 1

    _logger: logging.Logger
    _sock: socket.socket
    _remote_sock: socket.socket
    _remote_addr: Tuple[str, int]
    _multithreading: bool
    _buffer: queue.Queue
    _started: bool
    _endpoint_type: EndpointType

    def __init__(self, addr: Tuple[str, int] = ("127.0.0.1", 12000), remote_addr: Tuple[str, int] = None,
                 send_b_size: int = 65536, recv_b_size: int = 65536,
                 multithreading: bool = False):
        """Creates a new endpoint, one of a pair that is able to communicate with one another over a persistent
        stream in one-way fashion. Allows the transmission of generic objects in both synchronous and asynchronous
        fashion.

        :param addr: Address of endpoint.
        :param remote_addr: Address of remote endpoint to connect to.
        :param send_b_size: Underlying send buffer size of socket.
        :param recv_b_size: Underlying receive buffer size of socket.
        :param multithreading: Enables transparent multithreading for speedup.
        """
        self._logger = logging.getLogger(f"StreamEndpoint[{addr}]")
        self._logger.debug("Initializing endpoint...")

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, send_b_size)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_b_size)
        self._sock.bind(addr)
        self._remote_addr = remote_addr

        self._multithreading = multithreading
        self._buffer = queue.Queue()
        self._started = False
        self._logger.debug("Endpoint initialized.")

    def start(self, endpoint_type: EndpointType):
        """Starts the endpoint, either in dual-threaded fashion or as part of the main thread. By doing so, the two
        endpoints are connected and the datastream is opened.

        :param endpoint_type: Every endpoint of a stream is either a source or a sink.
        :raises RuntimeError: If already called for the endpoint.
        """
        self._logger.debug("Starting endpoint...")
        if self._started:
            raise RuntimeError(f"Endpoint already started ({self._endpoint_type.name})!")

        if endpoint_type.value == 0:
            if self._multithreading:
                threading.Thread(target=self._create_source).start()
            else:
                self._create_source()
        else:
            if self._multithreading:
                threading.Thread(target=self._create_sink).start()
            else:
                self._create_sink()

        self._started = True
        self._endpoint_type = endpoint_type
        self._logger.debug("Endpoint started.")

    def send(self, obj: object):
        """Generic send function that sends any object as a pickle over the persistent datastream. If multithreading is
        enabled, this function is asynchronous, otherwise it is blocking.

        :param obj: Object to send.
        """
        self._logger.debug("Sending object...")
        if not self._started:
            raise RuntimeError("Endpoint has not been started!")
        elif self._endpoint_type.value == 1:
            raise RuntimeError("Endpoint is not a source!")

        p_data = pickle.dumps(obj)

        if self._multithreading:
            self._logger.debug("Multithreading detected, putting object into internal buffer.")
            self._buffer.put(p_data)
        else:
            try:
                self._send(p_data)
            except TimeoutError:
                self._logger.debug("Timeout while trying to send object. Retrying...")
                while True:
                    try:
                        self._sock.connect(self._remote_addr)
                        self._send(p_data)
                        break
                    except TimeoutError:
                        continue
            self._logger.debug(f"Pickled object sent of size {len(p_data)}.")

    def recv(self) -> object:
        """Generic receive function that receives data as a pickle over the persistent datastream, unpickles it into the
         respective object and returns it. Blocking.

        :return: Received object.
        """
        self._logger.debug("Receiving object...")
        if not self._started:
            raise RuntimeError("Endpoint has not been started!")
        elif self._endpoint_type.value == 0:
            raise RuntimeError("Endpoint is not a sink!")

        if self._multithreading:
            self._logger.debug(
                f"Multithreading detected, retrieving object from internal buffer (size={self._buffer.qsize()}).")
            p_data = self._buffer.get()
        else:
            p_data = self._recv()

        self._logger.debug(f"Pickled object received of size {len(p_data)}.")
        return pickle.loads(p_data)

    def _create_source(self):
        self._logger.debug("Creating source...")
        self._sock.connect(self._remote_addr)
        self._logger.debug("Connected to remote endpoint.")

        if self._multithreading:
            self._logger.debug("Multithreading enabled.")
            while True:
                self._logger.debug(
                    f"Retrieving object to send from internal buffer (size={self._buffer.qsize()})...")
                p_data = self._buffer.get()
                try:
                    self._send(p_data)
                except TimeoutError:
                    self._logger.debug("Timeout while trying to send object. Retrying...")
                    while True:
                        try:
                            self._sock.connect(self._remote_addr)
                            self._send(p_data)
                            break
                        except TimeoutError:
                            continue
                self._logger.debug(f"Pickled object sent of size {len(p_data)}.")

    def _create_sink(self):
        self._logger.debug("Creating sink...")
        self._sock.listen()
        self._remote_sock, self._remote_addr = self._sock.accept()
        self._logger.debug("Connected to remote endpoint.")

        if self._multithreading:
            self._logger.debug("Multithreading enabled.")
            while True:
                try:
                    p_data = self._recv()
                    self._buffer.put(p_data)
                    self._logger.debug(f"Data received of size {len(p_data)}.")
                except TimeoutError:  # FIXME
                    self._logger.debug("Timeout while trying to receive data. Retrying...")
                    self._remote_sock, self._remote_addr = self._sock.accept()

    # noinspection PyTypeChecker
    def _send(self, data: bytes):
        def send_data(payload, size):
            sent_bytes = 0
            while sent_bytes < size:
                sent_bytes += self._sock.send(payload[sent_bytes:])

        data_size = len(data)
        p_data_size = bytes(ctypes.c_uint32(data_size))
        send_data(p_data_size, 4)
        send_data(data, data_size)

    def _recv(self) -> bytes:
        def recv_data(size, buff_size=4096):
            data = b''
            while size > 0:
                part = self._remote_sock.recv(min(size, buff_size))
                data += part
                size -= len(part)
            return data

        p_data_size = recv_data(4, 1)
        data_size = int.from_bytes(p_data_size, byteorder='little')
        return recv_data(data_size)
