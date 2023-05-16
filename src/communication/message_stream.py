"""
    An efficient, persistent, and stateless communications stream between two endpoints over BSD sockets. Supports SSL
    and LZ4 compression.

    Author: Fabian Hofmann
    Modified: 10.05.23
"""

import ctypes
import logging
import pickle
import queue
import socket
import sys
import threading
from time import sleep
from typing import Callable, Optional

from lz4.frame import compress, decompress


# TODO optional SSL https://docs.python.org/3/library/ssl.html
# TODO duplex (only one type of endpoint that is both able to receive and send)
# TODO support exclusionary server that only wants to re-connect to specific client
# TODO n:m endpoints
# TODO nested function optimization

class EndpointSocket:
    """A bundle of up to two sockets, that is used to communicate with another endpoint over a persistent TCP
    connection in synchronous manner. Supports authentication and encryption over SSL, and stream compression using
    LZ4. Thread-safe. (TODO EXTEND SAFETY BLOCK)
    """
    _listen_socks: dict[tuple, tuple[socket.socket, dict[tuple, socket.socket], threading.Lock]] = {} # FIXME
    _listen_lock = threading.Lock()
    _acc_socks: list[tuple[tuple, socket.socket]] = [] # FIXME THIS SHOULD BE SORTED UNDER THE TUPLES OF LISTEN SOCKS
    _acc_addrs: set[tuple] = {} # FIXME THIS SHOULD BE SORTED UNDER THE TUPLES OF LISTEN SOCKS
    _acc_lock = threading.Lock() # FIXME THIS SHOULD BE SORTED UNDER THE TUPLES OF LISTEN SOCKS

    _addr: tuple[str, int]
    _remote_addr: Optional[tuple[str, int]]
    _acceptor: bool
    _send_b_size: int
    _recv_b_size: int

    _sock: Optional[socket.socket]
    _sock_lock: threading.Lock

    _logger: logging.Logger
    _started: bool

    def __init__(self, addr: tuple[str, int], remote_addr: tuple[str, int] = None, acceptor: bool = True,
                 send_b_size: int = 65536, recv_b_size: int = 65536):
        """Creates a new socket endpoint.

        :param addr: Address of endpoint.
        :param remote_addr: Address of remote endpoint to connect to if in client mode. # FIXME
        :param acceptor: TODO
        :param send_b_size: Underlying send buffer size of socket.
        :param recv_b_size: Underlying receive buffer size of socket.
        """
        self._addr = addr
        self._remote_addr = remote_addr
        self._acceptor = acceptor
        self._send_b_size = send_b_size
        self._recv_b_size = recv_b_size

        self._sock = None
        self._sock_lock = threading.Lock()

        self._logger = logging.getLogger()
        self._started = False

    def start(self, timeout: int = None):  # FIXME
        self._started = True
        with self._sock_lock:
            self._open(timeout=timeout)

    def stop(self):  # FIXME THREADING
        self._started = False
        with self._sock_lock:
            self._close()

    def send(self, p_data: bytes):
        """Sends the given bytes of a single object over the connection, performing simple marshalling (size is
        sent first, then the bytes of the object). Fault-tolerant for breakdowns and resets in the connection.
        Blocking.

        :param p_data: Bytes to send.
        """
        while self._started:
            try:
                with self._sock_lock:
                    _send_payload(self._sock, p_data)
                return
            except (OSError, RuntimeError) as e:
                self._logger.info(f"{e.__class__.__name__}({e}) while trying to send data. Retrying...")
                with self._sock_lock:
                    self._open()

    def recv(self, timeout: int = None) -> bytes:
        """Receives the bytes of a single object sent over the connection, performing simple marshalling (size is
        sent first, then the bytes of the object). Fault-tolerant for breakdowns and resets in the connection.
        Blocking in default-mode if timeout not set.

        :param timeout: Timeout (seconds) to receive an object to return.
        :return: Received bytes.
        :raises TimeoutError: If timeout set and triggered.
        """
        while self._started:
            try:
                with self._sock_lock:
                    self._sock.settimeout(timeout)
                    p_data = _recv_payload(self._sock)
                    self._sock.settimeout(None)
                return p_data
            except (OSError, RuntimeError) as e:
                if timeout is not None and type(e) is TimeoutError:
                    raise e
                self._logger.info(f"{e.__class__.__name__}({e}) while trying to receive data. Retrying...")
                with self._sock_lock:
                    self._open()

    def _open(self):  # FIXME BLOCK
        """Establishes a connection to the remote socket, opening the socket first and initializing the connection
        socket if in server mode. Fault-tolerant for breakdowns and resets in the connection. FIXME
        """  # FIXME NAME
        while self._started:
            try:
                self._close()
                if self._acceptor:
                    l_sock, l_sock_lock = self._get_l_socket()  # FIXME BLOCK
                    self._accept(l_sock, l_sock_lock)
                else:
                    self._sock = self._get_c_socket()  # FIXME BLOCK
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self._send_b_size)
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._recv_b_size)
                return
            except (OSError, RuntimeError) as e:
                self._logger.info(f"{e.__class__.__name__}({e}) while trying to establish connection. Retrying...")
                sleep(1)
                continue

    def _accept(self, l_sock: socket.socket, c_socks: dict[tuple, socket.socket], l_sock_lock: threading.Lock): # FIXME BLOCK
        """

        :param l_sock:
        :param l_sock_lock:
        """
        while self._started:
            with l_sock_lock:  # FIXME
                a_sock = c_socks.pop(self._remote_addr, None)
            if a_sock is not None:
                self._sock = a_sock
                break
            else:
                with EndpointSocket._acc_lock:  # FIXME
                    a_sock = EndpointSocket._acc_socks.pop(self._remote_addr, None)

                with l_sock_lock:
                    a_sock, a_remote_addr = l_sock.accept()
                if self._remote_addr is None or self._remote_addr == a_remote_addr:
                    self._sock = a_sock
                    self._remote_addr = a_remote_addr
                    break
                else:
                    with EndpointSocket._acc_lock:
                        EndpointSocket._acc_socks.append((a_remote_addr, a_sock))
                    continue

    def _close(self):
        """Closes the socket of an endpoint, shutdowns any potential connection that might have been established.
        """  # FIXME check global structures to clean
        _close_socket(self._sock)

    def __del__(self):
        if self._started:
            self.stop()

    @classmethod
    def _get_l_socket(cls, addr: tuple[str, int], remote_addr: tuple[str, int] = None) -> socket.socket:
        """Opens the main socket, binding it to a functioning address and if in client mode (remote_addr is not
        None), also tries to establish a connection to the remote socket. Supports hostname resolution. FIXME

        :param addr:
        :param remote_addr:
        :raises RuntimeError: If none of the addresses succeed to create a working socket.
        :return:
        """
        for res in socket.getaddrinfo(*addr, type=socket.SOCK_STREAM):
            s_af, s_t, s_p, _, s_addr = res
            l_name = socket.getnameinfo(s_addr, socket.NI_NUMERICSERV)
            with cls._listen_lock:
                l_sock_t = cls._listen_socks.get(l_name, None)
                if l_sock_t is not None:
                    l_sock, c_socks, l_sock_lock = l_sock_t
                else:
                    try:
                        l_sock = socket.socket(s_af, s_t, s_p)
                        l_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        l_sock.bind(s_addr)
                        l_sock.listen(65535)
                    except OSError:
                        _close_socket(l_sock)
                        continue
                    l_sock_lock = threading.Lock()
                    c_socks = {}
                    cls._listen_socks[l_name] = l_sock, c_socks, l_sock_lock
            return l_sock, c_socks, l_sock_lock
        raise RuntimeError(f"Could not open socket with address ({addr}, {remote_addr})")

    @classmethod
    def _get_c_socket(cls, addr: tuple[str, int], remote_addr: tuple[str, int] = None) -> socket.socket:
        """Opens the main socket, binding it to a functioning address and if in client mode (remote_addr is not
        None), also tries to establish a connection to the remote socket. Supports hostname resolution. FIXME

        :param addr:
        :param remote_addr:
        :raises RuntimeError: If none of the addresses succeed to create a working socket.
        :return:
        """
        for res in socket.getaddrinfo(*addr, type=socket.SOCK_STREAM):
            s_af, s_t, s_p, _, s_addr = res
            r_res_list = socket.getaddrinfo(*remote_addr, family=s_af.value, type=s_t.value, proto=s_p)
            for r_res in r_res_list:
                r_af, r_t, r_p, _, r_addr = r_res
                sock = None
                try:
                    sock = socket.socket(r_af, r_t, r_p)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(s_addr)
                    sock.connect(r_addr)  # FIXME BLOCKING
                except OSError:
                    _close_socket(sock)
                    continue
                return sock
        raise RuntimeError(f"Could not open socket with address ({addr}, {remote_addr})")


class StreamEndpoint:
    """One of a pair of endpoints that is able to communicate with one another over a persistent stateless stream over
    BSD sockets. Allows the transmission of generic objects in both synchronous and asynchronous fashion. Supports SSL
    and LZ4 compression for the stream.
    """
    _logger: logging.Logger

    _endpoint_socket: EndpointSocket
    _marshal_f: Callable[[object], bytes]
    _unmarshal_f: Callable[[bytes], object]

    _multithreading: bool
    _send_thread: threading.Thread
    _recv_thread: threading.Thread
    _send_buffer: queue.Queue
    _recv_buffer: queue.Queue
    _started: bool

    def __init__(self, addr: tuple[str, int] = ("127.0.0.1", 12000), remote_addr: tuple[str, int] = None,
                 server_mode: bool = True, send_b_size: int = 65536, recv_b_size: int = 65536,
                 compression: bool = False, marshal_f: Callable[[object], bytes] = pickle.dumps,
                 unmarshal_f: Callable[[bytes], object] = pickle.loads,
                 multithreading: bool = False, buffer_size: int = 1024):
        """Creates a new endpoint.

        :param addr: Address of endpoint.
        :param remote_addr: Address of remote endpoint to connect to.
        :param server_mode: TODO
        :param send_b_size: Underlying send buffer size of socket.
        :param recv_b_size: Underlying receive buffer size of socket.
        :param compression: Enables lz4 stream compression for bandwidth optimization.
        :param marshal_f: Marshal function to serialize objects to send into bytes.
        :param unmarshal_f: Unmarshal function to deserialize received bytes into objects.
        :param multithreading: Enables transparent multithreading for speedup.
        :param buffer_size: Size of shared buffer in multithreading mode.
        """
        self._logger = logging.getLogger()
        self._logger.info("Initializing endpoint...")

        self._endpoint_socket = EndpointSocket(addr, remote_addr, server_mode, send_b_size, recv_b_size)

        if compression:
            self._marshal_f = lambda d: compress(marshal_f(d))
            self._unmarshal_f = lambda e: unmarshal_f(decompress(e))
        else:
            self._marshal_f = marshal_f
            self._unmarshal_f = unmarshal_f

        self._multithreading = multithreading
        self._send_buffer = queue.Queue(maxsize=buffer_size)
        self._recv_buffer = queue.Queue(maxsize=buffer_size)
        self._started = False
        self._logger.info("Endpoint initialized.")

    def start(self, timeout: int = None):
        """Starts the endpoint, either in dual-threaded fashion or as part of the main thread. By doing so, the two
        endpoints are connected and the datastream is opened.

        :param timeout: Timeout (seconds) to receive an object to return.
        :raises RuntimeError: If endpoint has already been started.
        :raises TimeoutError: If timeout set and triggered.
        """
        self._logger.info("Starting endpoint...")
        if self._started:
            raise RuntimeError(f"Endpoint has already been started!")
        self._started = True
        self._endpoint_socket.start(timeout=timeout)

        if self._multithreading:
            self._logger.info("Multithreading detected, starting endpoint sender/receiver threads...")
            self._send_thread = threading.Thread(target=self._create_sender, daemon=True)
            self._recv_thread = threading.Thread(target=self._create_receiver, daemon=True)
            self._send_thread.start()
            self._recv_thread.start()
        self._logger.info("Endpoint started.")

    def stop(self):
        """Stops the endpoint and closes the stream, cleaning up underlying structures. If multithreading is enabled,
        waits until the endpoint thread stops before performing cleanup. Note this does not guarantee the sending and
        receiving of all objects still pending --- they may still be in internal buffers and will be processed if the
        endpoint is opened again, or get discarded by the underlying socket.

        :raises RuntimeError: If endpoint has not been started.
        """
        self._logger.info("Stopping endpoint...")
        if not self._started:
            raise RuntimeError(f"Endpoint has not been started!")
        self._started = False

        if self._multithreading:
            self._logger.info("Multithreading detected, waiting for endpoint sender/receiver threads to stop...")
            self._send_thread.join()
            self._recv_thread.join()
        self._endpoint_socket.stop()
        self._logger.info("Endpoint stopped.")

    def send(self, obj: object):
        """Generic send function that sends any object as a pickle over the persistent datastream. If multithreading is
        enabled, this function is asynchronous, otherwise it is blocking.

        :param obj: Object to send.
        :raises RuntimeError: If endpoint has not been started.
        """
        self._logger.debug("Sending object...")
        if not self._started:
            raise RuntimeError("Endpoint has not been started!")

        p_obj = self._marshal_f(obj)
        if self._multithreading:
            self._logger.debug(
                f"Multithreading detected, putting object into buffer (size={self._send_buffer.qsize()})...")
            self._send_buffer.put(p_obj)
        else:
            self._endpoint_socket.send(p_obj)
        self._logger.debug(f"Pickled object sent of size {len(p_obj)}.")

    def receive(self, timeout: int = None) -> object:
        """Generic receive function that receives data as a pickle over the persistent datastream, unpickles it into the
         respective object and returns it. Blocking in default-mode if timeout not set.

        :param timeout: Timeout (seconds) to receive an object to return.
        :return: Received object.
        :raises RuntimeError: If endpoint has not been started.
        :raises TimeoutError: If timeout set and triggered.
        """
        self._logger.debug("Receiving object...")
        if not self._started:
            raise RuntimeError("Endpoint has not been started!")

        if self._multithreading:
            self._logger.debug(
                f"Multithreading detected, retrieving object from buffer (size={self._recv_buffer.qsize()})...")
            try:
                p_obj = self._recv_buffer.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError
        else:
            p_obj = self._endpoint_socket.recv(timeout)
        self._logger.debug(f"Pickled data received of size {len(p_obj)}.")
        return self._unmarshal_f(p_obj)

    def _create_sender(self):
        """Starts the loop to send objects over the socket retrieved from the sending buffer.
        """
        self._logger.info(f"AsyncSender: Starting to send objects in asynchronous mode...")
        while self._started:
            try:
                p_obj = self._send_buffer.get(timeout=10)
                self._logger.debug(
                    f"AsyncSender: Retrieved and sending object (size: {len(p_obj)}) from buffer "
                    f"(length: {self._send_buffer.qsize()})")
                self._endpoint_socket.send(p_obj)
            except queue.Empty:
                self._logger.warning(f"AsyncSender: Timeout triggered: Buffer empty. Retrying...")
        self._logger.info(f"AsyncSender: Stopping...")

    def _create_receiver(self):
        """Starts the loop to receive objects over the socket and store them in the receiving buffer.
        """
        self._logger.info(f"AsyncReceiver: Starting to receive objects in asynchronous mode...")
        while self._started:
            try:
                p_obj = self._endpoint_socket.recv()
                self._logger.debug(
                    f"AsyncReceiver: Storing received object (size: {len(p_obj)}) in buffer "
                    f"(length: {self._recv_buffer.qsize()})...")
                self._recv_buffer.put(p_obj, timeout=10)
            except queue.Full:
                self._logger.warning(f"AsyncReceiver: Timeout triggered: Buffer full. Discarding object...")
        self._logger.info(f"AsyncReceiver: Stopping...")

    def __iter__(self):
        while self._started:
            yield self.receive()

    def __del__(self):
        if self._started:
            self.stop()


# noinspection PyTypeChecker
def _send_payload(sock: socket.socket, payload: bytes):
    """TODO

    :param sock:
    :param payload:
    """
    payload_size = len(payload)
    p_payload_size = bytes(ctypes.c_uint32(payload_size))
    _send_n_data(sock, p_payload_size, 4)
    _send_n_data(sock, payload, payload_size)


def _send_n_data(sock: socket.socket, data: bytes, size: int):
    """TODO

    :param sock:
    :param data:
    :param size:
    """
    sent_bytes = 0
    while sent_bytes < size:
        n_sent_bytes = sock.send(data[sent_bytes:])
        if n_sent_bytes == 0:
            raise RuntimeError("Connection broken!")
        sent_bytes += n_sent_bytes


def _recv_payload(sock: socket.socket) -> bytes:
    """TODO

    :param sock:
    :return:
    """
    p_payload_size = _recv_n_data(sock, 4, 1)
    payload_size = int.from_bytes(p_payload_size, byteorder=sys.byteorder)
    return _recv_n_data(sock, payload_size)


def _recv_n_data(sock: socket.socket, size: int, buff_size: int = 4096) -> bytes:
    """TODO

    :param sock:
    :param size:
    :param buff_size:
    :return:
    """
    data = bytearray(size)
    r_size = size
    while r_size > 0:
        n_data = sock.recv(min(r_size, buff_size))
        n_size = len(n_data)
        if n_size == 0:
            raise RuntimeError("Connection broken!")
        data[size - r_size:size - r_size + n_size] = n_data
        r_size -= n_size
    return data


def _close_socket(sock: socket.socket):
    """Closes the socket of an endpoint, shutdowns any potential connection that might have been established.
    :param sock:
    :return:
    """  # TODO
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()
