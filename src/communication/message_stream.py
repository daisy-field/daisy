import ctypes
import logging
import pickle
import queue
import socket
import threading
from enum import Enum
from time import sleep
from typing import Tuple


# TODO set up SSL https://docs.python.org/3/library/ssl.html
# TODO filter logging
# TODO filter exceptions
# TODO fix misc FIXME
# TODO docstrings, typehints

class StreamEndpoint:
    class EndpointType(Enum):
        SOURCE = 0
        SINK = 1

    _logger: logging.Logger
    _endpoint_type: EndpointType
    _sock: socket.socket
    _remote_sock: socket.socket
    _remote_addr: Tuple[str, int]
    _multithreading: bool
    _buffer: queue.Queue
    _started: bool

    def __init__(self, addr: Tuple[str, int] = ("127.0.0.1", 12000), remote_addr: Tuple[str, int] = None,
                 endpoint_type: EndpointType = EndpointType.SINK, send_b_size: int = 65536, recv_b_size: int = 65536,
                 multithreading: bool = False):
        """Creates a new endpoint, one of a pair that is able to communicate with one another over a persistent
        stream in one-way fashion over BSD sockets. Allows the transmission of generic objects in both synchronous and
        asynchronous fashion.

        :param addr: Address of endpoint.
        :param remote_addr: Address of remote endpoint to connect to.
        :param endpoint_type: Every endpoint of a stream is either a source or a sink.
        :param send_b_size: Underlying send buffer size of socket.
        :param recv_b_size: Underlying receive buffer size of socket.
        :param multithreading: Enables transparent multithreading for speedup.
        """
        self._logger = logging.getLogger(f"StreamEndpoint[{addr}]")
        self._logger.debug("Initializing endpoint...")

        self._endpoint_type = endpoint_type
        if endpoint_type.value == 0 and remote_addr is None:
            raise ValueError("Endpoint of type source requires a remote address to pair to!")

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, send_b_size)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_b_size)
        self._sock.bind(addr)
        self._remote_addr = remote_addr

        self._multithreading = multithreading
        self._buffer = queue.Queue(maxsize=1024)
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
            raise RuntimeError(f"Endpoint has already been started!")

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
        self._logger.debug("Endpoint started.")

    def stop(self):  # FIXME
        self._logger.debug("Stopping endpoint...")
        if not self._started:
            self._logger.debug("Endpoint has not been started!")
            return

        self._started = False

        if self._endpoint_type.value == 1:
            self._remote_sock.shutdown(socket.SHUT_RDWR)
            self._remote_sock.close()
        self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()

        self._logger.debug("Endpoint stopped.")

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

        p_obj = pickle.dumps(obj)

        if self._multithreading:
            self._logger.debug(
                f"Multithreading detected, putting object into internal buffer (size={self._buffer.qsize()})...")
            self._buffer.put(p_obj)
        else:
            self._send(p_obj)
        self._logger.debug(f"Pickled object sent of size {len(p_obj)}.")

    def receive(self) -> object:
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
                f"Multithreading detected, retrieving object from internal buffer (size={self._buffer.qsize()})...")
            p_obj = self._buffer.get()
        else:
            p_obj = self._recv()

        self._logger.debug(f"Pickled object received of size {len(p_obj)}.")
        return pickle.loads(p_obj)

    def _create_source(self):
        self._logger.debug("Creating source...")
        self._connect()
        self._logger.debug("Connected to remote endpoint.")

        if self._multithreading:
            self._logger.debug("Multithreading enabled.")
            while self._started:  # FIXME
                self._logger.debug(
                    f"Retrieving object to send from internal buffer (size={self._buffer.qsize()})...")
                p_data = self._buffer.get()
                self._send(p_data)
                self._logger.debug(f"Pickled object sent of size {len(p_data)}.")

    def _create_sink(self):
        self._logger.debug("Creating sink...")
        self._sock.listen()
        self._accept()
        self._logger.debug("Connected to remote endpoint.")

        if self._multithreading:
            self._logger.debug("Multithreading enabled.") # FIXME
            while self._started:  # FIXME
                p_data = self._recv()
                self._buffer.put(p_data)
                self._logger.debug(f"Data received of size {len(p_data)}.") # FIXME

    def _connect(self):
        while self._started:
            try:
                self._sock.connect(self._remote_addr)
                break
            except TimeoutError:  # FIXME filter errors, better outputs
                self._logger.debug("Retrying...")  # FIXME
                sleep(1)
                continue

    def _accept(self):
        while self._started:
            try:
                self._remote_sock, self._remote_addr = self._sock.accept()
            except TimeoutError:  # FIXME filter errors, better outputs
                self._logger.debug("Retrying...")  # FIXME
                self._remote_sock.shutdown(socket.SHUT_RDWR)
                self._remote_sock.close()
                sleep(1)
                continue

    def _send(self, p_data: bytes) -> None:
        # noinspection PyTypeChecker
        def send_payload(payload):
            def send_n_data(data, size):
                sent_bytes = 0
                while sent_bytes < size:
                    n_sent_bytes = self._sock.send(data[sent_bytes:])
                    if n_sent_bytes == 0:
                        raise RuntimeError("Stream broken!")
                    sent_bytes += n_sent_bytes

            payload_size = len(payload)
            p_payload_size = bytes(ctypes.c_uint32(payload_size))
            send_n_data(p_payload_size, 4)
            send_n_data(payload, payload_size)

        while self._started:
            try:
                return send_payload(p_data)
            except TimeoutError:  # FIXME RuntimeError BrokenPipeError, ConnectionRefusedError
                self._logger.debug("Timeout while trying to send data. Retrying...")  # FIXME
                self._connect()

    def _recv(self) -> bytes:
        def recv_payload():
            def recv_n_data(size, buff_size=4096):
                data = b''
                while size > 0:
                    n_data = self._remote_sock.recv(min(size, buff_size))
                    if n_data == b'':
                        raise RuntimeError("Stream broken!")
                    data += n_data
                    size -= len(n_data)
                return data

            p_payload_size = recv_n_data(4, 1)
            payload_size = int.from_bytes(p_payload_size, byteorder='little')
            return recv_n_data(payload_size)

        while self._started:
            try:
                return recv_payload()
            except TimeoutError:  # FIXME
                self._logger.debug("Timeout while trying to receive data. Retrying...")  # FIXME
                self._accept()

    def __iter__(self):
        while self._started:
            yield self.receive()

    def __del__(self):
        self.stop()
