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
        """TODO

        :param addr: Address of endpoint.
        :param remote_addr: Address of remote endpoint to connect to.
        :param send_b_size: Underlying send buffer size of socket.
        :param recv_b_size: Underlying receive buffer size of socket.
        :param multithreading: Enables transparent multithreading for speedup.
        """
        self._logger = logging.getLogger(f"StreamEndpoint[{addr}]")

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(addr)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, send_b_size)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_b_size)

        self._remote_addr = remote_addr
        self._multithreading = multithreading

        self._started = False

    def start(self, endpoint_type: EndpointType):
        """TODO

        :param endpoint_type:
        :raises RuntimeError:
        """
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

    def send(self, obj):
        """TODO

        :param obj:
        """
        p_data = pickle.dumps(obj)

        if self._multithreading:
            self._buffer.put(p_data)
        else:
            try:
                self._send(p_data)
            except TimeoutError:
                while True:
                    try:
                        self._sock.connect(self._remote_addr)
                        self._send(p_data)
                        break
                    except TimeoutError:
                        continue

    def recv(self) -> object:
        """TODO

        :return:
        """
        if self._multithreading:
            p_data = self._buffer.get()
        else:
            p_data = self._recv()
        return pickle.loads(p_data)

    def _create_source(self):
        self._sock.connect(self._remote_addr)

        if self._multithreading:
            while True:
                p_data = self._buffer.get()
                try:
                    self._send(p_data)
                except TimeoutError:
                    while True:
                        try:
                            self._sock.connect(self._remote_addr)
                            self._send(p_data)
                            break
                        except TimeoutError:
                            continue

    def _create_sink(self):
        self._sock.listen()
        self._remote_sock, self._remote_addr = self._sock.accept()

        if self._multithreading:
            while True:
                try:
                    p_data = self._recv()
                    self._buffer.put(p_data)
                except TimeoutError:
                    self._remote_sock, self._remote_addr = self._sock.accept()

    # noinspection PyTypeChecker
    def _send(self, data):
        def send_data(payload, size):
            sent_bytes = 0
            while sent_bytes < size:
                sent_bytes += self._remote_sock.send(payload[sent_bytes:])

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
