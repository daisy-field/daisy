import ctypes
import logging
import pickle
import queue
import socket
import threading
from enum import Enum
from time import sleep
from typing import Optional
from typing import Tuple


# TODO set up SSL https://docs.python.org/3/library/ssl.html
# TODO checkout gzip/lzma/bz2/mgzip/zipfile for compression
# TODO filter logging
# TODO int conversions

class StreamEndpoint:
    class EndpointType(Enum):
        SOURCE = 0
        SINK = 1

    class EndpointSocket:
        _addr: Tuple[str, int]
        _remote_addr: Optional[Tuple[str, int]]
        _send_b_size: int
        _recv_b_size: int
        _sock: Optional[socket.socket]
        _remote_sock: Optional[socket.socket]

        _logger: logging.Logger
        _started: bool

        def __init__(self, addr: Tuple[str, int], remote_addr: Tuple[str, int], send_b_size: int, recv_b_size: int,
                     logger: logging.Logger):
            """Creates a new socket endpoint, a bundle of up to two sockets, that is used to communicate with another
            endpoint over a persistent TCP connection.

            :param addr: Address of endpoint.
            :param remote_addr: Address of remote endpoint to connect to.
            :param send_b_size: Underlying send buffer size of socket.
            :param recv_b_size: Underlying receive buffer size of socket.
            :param logger: Logger to use.
            """
            self._addr = addr
            self._remote_addr = remote_addr
            self._send_b_size = send_b_size
            self._recv_b_size = recv_b_size

            self._logger = logger
            self._started = False

        def start(self):
            self._started = True

        def stop(self):
            self._started = False

        def connect(self):
            """Establishes a connection to the remote socket, opening the socket first and initializing the connection
            socket if in server mode (sink). Fault-tolerant for breakdowns and resets in the connection.
            """
            while self._started:
                try:
                    self.close()
                    self._open()
                    if self._remote_addr is None:
                        self._sock.listen(0)
                        self._remote_sock, _ = self._sock.accept()
                        self._remote_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._recv_b_size)
                    break
                except OSError as e:
                    self._logger.debug(f"{e.__class__.__name__}({e}) while trying to establish connection. Retrying...")
                    sleep(1)
                    continue

        def close(self):
            """Closes the sockets of an endpoint, shutdowns any potential connection that might have been established.
            """
            def close_socket(sock: socket.socket):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()

            if self._remote_addr is not None:
                close_socket(self._remote_sock)
            close_socket(self._sock)

        def send(self, p_data: bytes):
            """Sends the given bytes of a single object over the connection, performing simple marshalling (size is
            sent first, then the bytes of the object). Fault-tolerant for breakdowns and resets in the connection.
            Blocking.

            :param p_data: Bytes to send.
            """

            # noinspection PyTypeChecker
            def send_payload(payload):
                def send_n_data(data, size):
                    sent_bytes = 0
                    while sent_bytes < size:
                        n_sent_bytes = self._sock.send(data[sent_bytes:])
                        if n_sent_bytes == 0:
                            raise RuntimeError("Connection broken!")
                        sent_bytes += n_sent_bytes

                payload_size = len(payload)
                p_payload_size = bytes(ctypes.c_uint32(payload_size))  # FIXME BYTEORDER socket.htonl(x)
                send_n_data(p_payload_size, 4)
                send_n_data(payload, payload_size)

            while self._started:
                try:
                    return send_payload(p_data)
                except (OSError, RuntimeError) as e:
                    self._logger.debug(f"{e.__class__.__name__}({e}) while trying to send data. Retrying...")
                    self.connect()

        def recv(self, timeout: int = None) -> bytes:
            """Receives the bytes of a single object sent over the connection, performing simple marshalling (size is
            sent first, then the bytes of the object). Fault-tolerant for breakdowns and resets in the connection.
            Blocking in default-mode if timeout not set.

            :param timeout: Timeout (seconds) to receive an object to return.
            :return: Received bytes.
            :raises TimeoutError: If timeout set and triggered.
            """

            def recv_payload():
                def recv_n_data(size, buff_size=4096):
                    data = b''
                    while size > 0:
                        n_data = self._remote_sock.recv(min(size, buff_size))
                        if n_data == b'':
                            raise RuntimeError("Connection broken!")
                        data += n_data
                        size -= len(n_data)
                    return data

                p_payload_size = recv_n_data(4, 1)
                payload_size = int.from_bytes(p_payload_size, byteorder='little')  # FIXME BYTEORDER socket.ntohl(x)
                return recv_n_data(payload_size)

            while self._started:
                try:
                    self._remote_sock.settimeout(timeout)
                    p_data = recv_payload()
                    self._remote_sock.settimeout(None)
                    return p_data
                except (OSError, RuntimeError) as e:
                    if timeout is not None and type(e) is TimeoutError:
                        raise e
                    self._logger.debug(f"{e.__class__.__name__}({e}) while trying to receive data. Retrying...")
                    self.connect()

        def _open(self):
            """Opens the main socket, binding it to a functioning address and if in client mode (source), also tries to
            establish a connection to the remote socket. Supports hostname resolution.

            :raises RuntimeError: If none of the addresses succeed to create a working socket.
            """
            for res in socket.getaddrinfo(*self._addr, type=socket.SOCK_STREAM):
                s_af, s_t, s_p, _, s_addr = res

                if self._remote_addr is not None:
                    r_res_list = socket.getaddrinfo(*self._remote_addr, family=s_af.value, type=s_t.value, proto=s_p)
                else:
                    r_res_list = [res]

                for r_res in r_res_list:
                    r_af, r_t, r_p, _, r_addr = r_res
                    try:
                        self._sock = socket.socket(r_af, r_t, r_p)
                        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    except OSError:
                        self._sock = None
                        continue
                    try:
                        self._sock.bind(s_addr)
                    except OSError:
                        self._sock.close()
                        self._sock = None
                        continue
                    if self._remote_addr is not None:
                        try:
                            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self._send_b_size)
                            self._sock.connect(r_addr)
                        except OSError:
                            self._sock.close()
                            self._sock = None
                            continue
                    return
            raise RuntimeError(f"Could not open socket with address ({self._addr}, {self._remote_addr})")

        def __del__(self):
            self.close()

    _logger: logging.Logger

    _endpoint_type: EndpointType
    _endpoint_socket: EndpointSocket

    _multithreading: bool
    _thread: threading.Thread
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
        elif endpoint_type.value == 1 and remote_addr is not None:
            raise ValueError("Endpoint of type sink does not require a remote address to pair to!")

        self._endpoint_socket = StreamEndpoint.EndpointSocket(addr, remote_addr, send_b_size, recv_b_size, self._logger)

        self._multithreading = multithreading
        self._buffer = queue.Queue(maxsize=1024)
        self._started = False
        self._logger.debug("Endpoint initialized.")

    def start(self):
        """Starts the endpoint, either in dual-threaded fashion or as part of the main thread. By doing so, the two
        endpoints are connected and the datastream is opened.

        :raises RuntimeError: If already called for the endpoint.
        """
        self._logger.debug("Starting endpoint...")
        if self._started:
            raise RuntimeError(f"Endpoint has already been started!")
        self._started = True
        self._endpoint_socket.start()

        if self._endpoint_type.value == 0:
            if self._multithreading:
                self._thread = threading.Thread(target=self._create_source, name="AsyncSource")
                self._thread.start()
            else:
                self._create_source()
        else:
            if self._multithreading:
                self._thread = threading.Thread(target=self._create_sink, name="AsyncSink")
                self._thread.start()
            else:
                self._create_sink()

        self._logger.debug("Endpoint started.")

    def stop(self):
        """Stops the endpoint and closes the stream, cleaning up underlying structures. If multithreading is enabled,
        waits until the endpoint thread stops before performing cleanup.
        """
        self._logger.debug("Stopping endpoint...")
        if not self._started:
            self._logger.debug("Endpoint has not been started!")
            return
        self._started = False
        self._endpoint_socket.stop()

        if self._multithreading:
            self._thread.join()
        else:
            self._endpoint_socket.close()
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
            self._endpoint_socket.send(p_obj)
        self._logger.debug(f"Pickled object sent of size {len(p_obj)}.")

    def receive(self, timeout: int = None) -> object:
        """Generic receive function that receives data as a pickle over the persistent datastream, unpickles it into the
         respective object and returns it. Blocking in default-mode if timeout not set.

        :param timeout: Timeout (seconds) to receive an object to return.
        :return: Received object.
        :raises TimeoutError: If timeout set and triggered.
        """
        self._logger.debug("Receiving object...")
        if not self._started:
            raise RuntimeError("Endpoint has not been started!")
        elif self._endpoint_type.value == 0:
            raise RuntimeError("Endpoint is not a sink!")

        if self._multithreading:
            self._logger.debug(
                f"Multithreading detected, retrieving object from internal buffer (size={self._buffer.qsize()})...")
            try:
                p_obj = self._buffer.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError
        else:
            p_obj = self._endpoint_socket.recv(timeout)

        self._logger.debug(f"Pickled object received of size {len(p_obj)}.")
        return pickle.loads(p_obj)

    def _create_source(self):
        """Creates the streaming endpoint as a source, i.e. an endpoint that is able to send messages. By doing so, two
        endpoints are connected and the datastream is opened. If started in multithreading mode, also starts the loop
        to send objects from the internal buffer.
        """
        self._logger.debug("Creating source...")
        self._endpoint_socket.connect()
        self._logger.debug("Connected to remote endpoint.")

        if self._multithreading:
            self._logger.debug("Multithreading enabled.")
            while self._started:
                try:
                    self._logger.debug(
                        f"Retrieving object to send from internal buffer (size={self._buffer.qsize()})...")
                    p_data = self._buffer.get(timeout=10)
                    self._endpoint_socket.send(p_data)
                    self._logger.debug(f"Pickled object sent of size {len(p_data)}.")
                except queue.Empty:
                    pass
            self._endpoint_socket.close()

    def _create_sink(self):
        """Creates the streaming endpoint as a sink, i.e. an endpoint that is able to receive messages. By doing so, two
        endpoints are connected and the datastream is opened. If started in multithreading mode, also starts the loop
        to receive objects and store them in the internal buffer.
        """
        self._logger.debug("Creating sink...")
        self._endpoint_socket.connect()
        self._logger.debug("Connected to remote endpoint.")

        if self._multithreading:
            self._logger.debug("Multithreading enabled.")
            while self._started:
                try:
                    p_data = self._endpoint_socket.recv()
                    self._buffer.put(p_data, timeout=10)
                    self._logger.debug(f"Data received of size {len(p_data)}.")
                except queue.Full:
                    pass
            self._endpoint_socket.close()

    def __iter__(self):
        while self._started:
            yield self.receive()

    def __del__(self):
        self.stop()
