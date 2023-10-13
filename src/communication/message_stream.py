"""
    An efficient, persistent, and stateless communications stream between two endpoints over BSD sockets. Supports SSL
    and LZ4 compression.

    Author: Fabian Hofmann
    Modified: 26.07.23

    TODO Future Work: SSL https://docs.python.org/3/library/ssl.html
    TODO Future Work: Poll & Select
    TODO Future Work: Generating new Endpoints through Generic Endpoints upon new connection
    TODO Future Work: Blocked remote addresses (and leaving l_sockets open)
    TODO - when socket is not shut down but only closed OR when never opened to begin with
"""

import ctypes
import logging
import pickle
import queue
import select
import socket
import sys
import threading
from time import sleep, time
from typing import Callable, Optional

from lz4.frame import compress, decompress


class EndpointSocket:
    """A bundle of up to two sockets, that is used to communicate with another endpoint over a persistent TCP
    connection in synchronous manner. Supports authentication and encryption over SSL, and stream compression using
    LZ4. Thread-safe for both access to the same endpoint socket and using multiple threads using endpoint sockets
    set to the same address (this is organized through an array of class variables, see below for more info).

    :cvar _listen_socks: Active listen sockets, along with a respective lock to access each safely.
    :cvar _acc_r_socks: Pending registered connection cache for each listen socket.
    :cvar _acc_p_socks: Pending unregistered connection queue for each listen socket.
    :cvar _reg_r_addrs: Registered remote addresses.
    :cvar _addr_map: Mapping between registered remote addresses and their aliases.
    :cvar _act_l_counts: Active thread counter for each listen socket. Socket closes if counter reaches zero.
    :cvar _lock: General purpose lock to ensure safe access to class variables.
    :cvar _cls_logger: General purpose logger for class methods.
    """
    _listen_socks: dict[tuple[str, int], tuple[socket.socket, threading.Lock]] = {}
    _acc_r_socks: dict[tuple[str, int], tuple[dict[tuple[str, int], socket.socket], threading.Lock]] = {}
    _acc_p_socks: dict[tuple[str, int], queue.Queue[tuple[socket.socket, tuple[str, int]]]] = {}
    _reg_r_addrs: set[tuple[str, int]] = set()
    _addr_map: dict[tuple[str, int], set[tuple[str, int]]] = {}
    _act_l_counts: dict[tuple[str, int], int] = {}
    _lock = threading.Lock()
    _cls_logger: logging.Logger = logging.getLogger("EndpointSocket")

    _logger: logging.Logger

    _addr: tuple[str, int]
    _remote_addr: Optional[tuple[str, int]]
    _acceptor: bool

    _send_b_size: int
    _recv_b_size: int
    _sock: Optional[socket.socket]
    _sock_lock: threading.Lock

    _opened: bool

    def __init__(self, name: str, addr: tuple[str, int], remote_addr: tuple[str, int] = None, acceptor: bool = True,
                 send_b_size: int = 65536, recv_b_size: int = 65536):
        """Creates a new endpoint socket. Implementation note: A pre-defined remote address is not a guarantee that this
        endpoint will successfully be allowed to initialize for this remote address --- for example if another endpoint
        sock with the same remote address (be it generic or pre-defined) has already been registered, then the current
        one will throw an error.

        :param name: Name of endpoint for logging purposes.
        :param addr: Address of endpoint.
        :param remote_addr: Address of remote endpoint to be connected to. Mandatory in initiator mode (acceptor set to
        false), for acceptor mode this fixes the remote endpoint that is allowed to be connected to this endpoint.
        :param acceptor: Determines whether the endpoint accepts or initiates connections to/from other endpoints.
        :param send_b_size: Underlying send buffer size of socket.
        :param recv_b_size: Underlying receive buffer size of socket.
        """
        self._logger = logging.getLogger(name + "-Socket")
        self._logger.info(f"Initializing endpoint socket {addr, remote_addr}...")

        self._addr = addr
        self._remote_addr = remote_addr
        self._acceptor = acceptor
        if acceptor and remote_addr is not None:
            self._reg_remote(remote_addr)
            self._fix_rp_acc_socks(addr, remote_addr)
        elif not acceptor and remote_addr is None:
            raise ValueError("Initiating endpoint socket requires a remote address!")

        self._send_b_size = send_b_size
        self._recv_b_size = recv_b_size
        self._sock = None
        self._sock_lock = threading.Lock()

        self._opened = False
        self._logger.info(f"Endpoint socket {addr, remote_addr} initialized.")

    def open(self):
        self._logger.info(f"Opening endpoint socket...")
        self._opened = True
        with self._sock_lock:
            if self._acceptor:
                self._open_l_socket(self._addr)
            self._connect()
        self._logger.info(f"Endpoint socket opened.")

    def close(self, shutdown: bool = False):
        self._logger.info(f"Closing endpoint socket...")
        self._opened = False
        with self._sock_lock:
            _close_socket(self._sock)
            self._sock = None
            if self._acceptor:
                if shutdown:
                    self._unreg_remote(self._addr, self._remote_addr)
                self._close_l_socket(self._addr)
        self._logger.info(f"Endpoint socket closed.")

    def send(self, p_data: bytes):
        """Sends the given bytes of a single object over the connection, performing simple marshalling (size is
        sent first, then the bytes of the object). Fault-tolerant for breakdowns and resets in the connection. Blocking.

        :param p_data: Bytes to send.
        """
        while self._opened:
            try:
                with self._sock_lock:
                    if self._sock is None:
                        continue
                    _send_payload(self._sock, p_data)
                return
            except (OSError, ValueError, RuntimeError) as e:
                self._logger.warning(f"{e.__class__.__name__}({e}) while trying to send data. Retrying...")
                with self._sock_lock:
                    self._connect()

    def recv(self, timeout: int = None) -> Optional[bytes]:
        """Receives the bytes of a single object sent over the connection, performing simple marshalling (size is
        received first, then the bytes of the object). Fault-tolerant for breakdowns and resets in the connection.
        Blocking in default-mode if timeout not set.

        :param timeout: Timeout (seconds) to receive an object to return.
        :return: Received bytes or None of end point socket has been closed.
        :raises TimeoutError: If timeout set and triggered.
        """
        while self._opened:
            try:
                with self._sock_lock:
                    if self._sock is None:
                        continue
                    if timeout is not None and not select.select([self._sock], [], [], timeout)[0]:
                        raise TimeoutError
                    elif not select.select([self._sock], [], [], 0)[0]:
                        continue
                    p_data = _recv_payload(self._sock)
                return p_data
            except (OSError, ValueError, RuntimeError) as e:
                if timeout is not None and type(e) is TimeoutError:
                    raise e
                self._logger.warning(f"{e.__class__.__name__}({e}) while trying to receive data. Retrying...")
                with self._sock_lock:
                    self._connect()
        return None

    def _connect(self):
        """(Re-)Establishes a connection to a/the remote endpoint socket, first performing any necessary cleanup of the
        underlying socket, before opening it again and trying to connect/accept a remote endpoint socket. Fault-tolerant
        for breakdowns and resets in the connection. Blocking.
        """
        while self._opened:
            try:
                self._logger.info(f"Trying to (re-)establish connection {self._addr, self._remote_addr}...")
                _close_socket(self._sock)
                if self._acceptor:
                    self._sock = None
                    remote_addr = self._remote_addr
                    while self._opened and self._sock is None:
                        self._sock, remote_addr = self._get_a_socket(self._addr, self._remote_addr)
                    self._remote_addr = remote_addr
                else:
                    self._sock = self._get_c_socket(self._addr, self._remote_addr)
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self._send_b_size)
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._recv_b_size)
                self._logger.info(f"Connection {self._addr, self._remote_addr} (re-)established.")
                break
            except (OSError, ValueError, AttributeError, RuntimeError) as e:
                self._logger.warning(f"{e.__class__.__name__}({e}) while trying to (re-)establish connection "
                                     f"{self._addr, self._remote_addr}. Retrying...")
                sleep(1)
                continue

    def __del__(self):
        if self._opened:
            self.close(shutdown=True)

    @classmethod
    def _reg_remote(cls, remote_addr: tuple[str, int]):
        """Registers a remote address into the class datastructures, notifying other endpoints of its existence. Tries
        to both resolve the address and finds its fully qualified hostname to reserve all its aliases. If only a single
        alias is already registered, aborts the whole registration process and registers none of the aliases.

        :param remote_addr: Remote address to register.
        :raises ValueError: If remote address is already registered (possibly by another caller).
        """
        cls._cls_logger.info(f"Registering remote address ({remote_addr})...")
        addr_mapping = set()
        for _, _, _, _, addr in socket.getaddrinfo(*remote_addr, type=socket.SOCK_STREAM):
            addr = _convert_addr_to_name(addr)
            addr_mapping.add(addr)

        cls._cls_logger.debug(f"Registering aliases of remote address ({remote_addr}): {addr_mapping}...")
        with cls._lock:
            if remote_addr in cls._addr_map:
                raise ValueError(f"Remote address ({remote_addr}) is already registered!")
            for addr in addr_mapping:
                if addr in cls._reg_r_addrs:
                    raise ValueError(f"Remote address ({addr}) (resolved from {remote_addr}) is already registered!")

            cls._addr_map[remote_addr] = addr_mapping
            for addr in addr_mapping:
                cls._reg_r_addrs.add(addr)
        cls._cls_logger.info(f"Remote address ({remote_addr}) registered.")

    @classmethod
    def _fix_rp_acc_socks(cls, addr: tuple[str, int], remote_addr: tuple[str, int]):
        """After a remote address has been registered, cycles the waiting connections of that remote address to the
        registered socket dictionary from the pending socket queue. Necessary, as new endpoint sockets may be created 
        while the listening socket is already opened and connections are already getting accepted.

        Note this method merely helps in speeding up the sorting of connections to the correct endpoint socket and
        operates in best-effort manner, as the whole method is not considered a critical section, i.e. during the
        cycling of connections, the registered connection could be accepted by another thread and be put into the
        pending connection cache, where it will be dequeued by another thread --- which will result in a ValueError
        since the remote peer is obviously registered. However, no much of an issue, as the error handling is done in
        the background during the handling of existing connections.
        
        :param addr: Address of endpoint.
        :param remote_addr: Remote address that was registered.
        """
        cls._cls_logger.info(
            f"Fixing registered and pending connection caches for address pair {addr, remote_addr}...")
        l_addr, _, _ = cls._get_l_socket(addr)
        with cls._lock:
            addr_mapping = cls._addr_map[remote_addr]
            acc_p_socks = cls._acc_p_socks[l_addr]
            acc_r_socks, acc_r_lock = cls._acc_r_socks[l_addr]

        p_socks = []
        while True:
            try:
                a_sock, a_addr = acc_p_socks.get_nowait()
                with acc_r_lock:
                    if a_addr in addr_mapping:
                        _close_socket(acc_r_socks.pop(a_addr, None))
                        acc_r_socks[a_addr] = a_sock
                    else:
                        p_socks.append((a_sock, a_addr))
            except queue.Empty:
                break
        for a_sock, a_addr in p_socks:
            try:
                acc_p_socks.put_nowait((a_sock, a_addr))
            except queue.Full:
                _close_socket(a_sock)
        cls._cls_logger.info(
            f"Registered and pending connection caches for address pair {addr, remote_addr} fixed.")

    @classmethod
    def _unreg_remote(cls, addr: tuple[str, int], remote_addr: tuple[str, int]):
        """Unregisters a remote address from the class datastructures. Uses the existing mappings from the original
        resolution. Also cycles any pending connections of that remote address from the registered socket dictionary to
        the pending socket queue so the connection may be accepted by any other (generic or pre-defined) endpoint
        socket listening on the same address.
        
        :param remote_addr: Remote address that was registered.
        """
        cls._cls_logger.info(f"Unregistering remote address pair {addr, remote_addr}...")
        l_addr, _, _ = cls._get_l_socket(addr)
        with cls._lock:
            addr_mapping = cls._addr_map.pop(remote_addr)
            acc_p_socks = cls._acc_p_socks[l_addr]
            acc_r_socks, acc_r_lock = cls._acc_r_socks[l_addr]

        for a_addr in addr_mapping:
            with cls._lock:
                cls._reg_r_addrs.discard(a_addr)
            with acc_r_lock:
                a_sock = acc_r_socks.pop(a_addr, None)
            if a_sock is not None:
                try:
                    acc_p_socks.put_nowait((a_sock, a_addr))
                except queue.Full:
                    _close_socket(a_sock)
        cls._cls_logger.info(f"Remote address pair {addr, remote_addr} unregistered.")

    @classmethod
    def _open_l_socket(cls, addr: tuple[str, int]):
        """Opens the socket listening to a given address, iff there are no further endpoint sockets listening on the
        same socket as well.

        :param addr: Address of listen socket to open.
        """
        cls._get_l_socket(addr, new_endpoint=True)

    @classmethod
    def _get_l_socket(cls, addr: tuple[str, int], new_endpoint: bool = False) \
            -> tuple[tuple[str, int], socket.socket, threading.Lock]:
        """Gets the socket listening to a given address. If this socket does not exist already, creates it and with it 
        all accompanying datastructures. Supports address resolution.

        :param addr: Address of listen socket.
        :raises RuntimeError: If none of the addresses/aliases succeed to create a working socket.
        :return: A tupel consisting of the address, the socket, and a lock to be used for accessing the socket.
        """
        cls._cls_logger.debug(f"Trying to retrieve listening socket for {addr}...")
        for res in socket.getaddrinfo(*addr, type=socket.SOCK_STREAM):
            s_af, s_t, s_p, _, s_addr = res
            l_addr = _convert_addr_to_name(s_addr)
            with cls._lock:
                l_sock, l_sock_lock = cls._listen_socks.get(l_addr, (None, None))
                if l_sock is None:
                    try:
                        l_sock = socket.socket(s_af, s_t, s_p)
                        l_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        l_sock.bind(s_addr)
                        l_sock.listen(65535)
                    except OSError:
                        _close_socket(l_sock)
                        continue
                    l_sock_lock = threading.Lock()
                    cls._listen_socks[l_addr] = l_sock, l_sock_lock
                    cls._acc_r_socks[l_addr] = {}, threading.Lock()
                    cls._acc_p_socks[l_addr] = queue.Queue(maxsize=512)
                if new_endpoint:
                    cls._act_l_counts[l_addr] = cls._act_l_counts.get(l_addr, 0) + 1
            cls._cls_logger.debug(f"Listening socket {l_addr, l_sock, l_sock_lock} for {addr} retrieved")
            return l_addr, l_sock, l_sock_lock
        raise RuntimeError(f"Could not open listen socket for {addr}!")

    @classmethod
    def _close_l_socket(cls, addr: tuple[str, int]):
        """Closes the socket listening to a given address, iff there are no further endpoint sockets listening on the 
        same socket as well.
        
        :param addr: Address of listen socket to close.
        """
        cls._cls_logger.debug(f"Performing cleanup for listening socket for {addr}...")
        l_addr, l_sock, l_sock_lock = cls._get_l_socket(addr)

        with cls._lock, l_sock_lock:
            cls._act_l_counts[l_addr] = cls._act_l_counts.get(l_addr, 0) - 1
            if cls._act_l_counts[l_addr] > 0:
                return
            cls._listen_socks.pop(l_addr)
            cls._acc_r_socks.pop(l_addr)
            cls._acc_p_socks.pop(l_addr)
            cls._act_l_counts.pop(l_addr)
            _close_socket(l_sock)
        cls._cls_logger.debug(f"Listening socket for {addr} closed.")

    @classmethod
    def _get_a_socket(cls, addr: tuple[str, int], remote_addr: tuple[str, int] = None) \
            -> tuple[Optional[socket.socket], Optional[tuple[str, int]]]:
        """Gets an accepted connection socket assigned to a socket of a given address. The connection socket can either
        be connected to an arbitrary endpoint or to a pre-defined remote peer (remote address). If remote address is not
        pre-defined, also registers the remote peer's address. As the underlying datastructures are shared between all
        endpoint sockets (of a process), a connection socket can either be retrieved from them or directly from the
        listen socket.

        :param addr: Address of listen socket.
        :param remote_addr: Address of remote endpoint to be connected to.
        :raises RuntimeError: If none of the addresses/aliases of the listen socket succeed to get a working socket.
        :return: Tuple of the connection socket and the address of the remote peer.
        """
        cls._cls_logger.debug(f"Trying to retrieve accept socket for {addr, remote_addr}...")
        l_addr, _, _ = cls._get_l_socket(addr)

        # Check the active connection cache first, as another Endpoint
        # might have accepted this one's registered connection already.
        if remote_addr is not None:
            a_sock = cls._get_r_acc_sock(l_addr, remote_addr)
            if a_sock is not None:
                return a_sock, remote_addr

        # Check the pending connection queue, if this thread does not care
        # about the address of the remote peer. If connection, registers it.
        else:
            a_sock, a_addr = cls._get_p_acc_sock(l_addr)
            if a_sock is not None:
                try:
                    cls._reg_remote(a_addr)
                    return a_sock, a_addr
                except ValueError:
                    _close_socket(a_sock)

        # Check the OS connection backlog for pending connections
        a_sock, a_addr = cls._get_n_acc_sock(l_addr, remote_addr)
        if a_sock is not None:
            if remote_addr is None:
                try:
                    cls._reg_remote(a_addr)
                    return a_sock, a_addr
                except ValueError:
                    _close_socket(a_sock)
            else:
                return a_sock, remote_addr
        return None, None

    @classmethod
    def _get_r_acc_sock(cls, l_addr: tuple[str, int], remote_addr: tuple[str, int]) -> Optional[socket.socket]:
        """Retrieves and returns a registered (accepted) connection socket assigned to a socket of a given address, if
        it exists in the (shared) active registered connection cache.

        :param l_addr: Address of listen socket.
        :param remote_addr: Address of remote endpoint to be connected to.
        :return: Tuple of the registered connection socket and the address of the remote peer.
        """
        cls._cls_logger.debug(
            f"Trying to retrieve accept socket for {l_addr, remote_addr} from registered connection cache...")
        with cls._lock:
            acc_r_socks, acc_r_lock = cls._acc_r_socks[l_addr]
        for _, _, _, _, addr in socket.getaddrinfo(*remote_addr, type=socket.SOCK_STREAM):
            addr = _convert_addr_to_name(addr)
            with acc_r_lock:
                a_sock = acc_r_socks.pop(addr, None)
            if a_sock is not None:
                return a_sock
        return None

    @classmethod
    def _get_p_acc_sock(cls, l_addr: tuple[str, int]) -> tuple[Optional[socket.socket], Optional[tuple[str, int]]]:
        """Retrieves and returns a pending, not registered (accepted) connection socket assigned to a socket of a
        given address, if there is one in the pending connection queue.

        :param l_addr: Address of listen socket.
        :return: Tuple of a connection socket and the address of the remote peer.
        """
        cls._cls_logger.debug(f"Trying to retrieve accept socket for {l_addr} from pending connection queue...")
        with cls._lock:
            acc_p_socks = cls._acc_p_socks[l_addr]
        try:
            return acc_p_socks.get_nowait()
        except queue.Empty:
            return None, None

    @classmethod
    def _get_n_acc_sock(cls, l_addr: tuple[str, int], remote_addr: tuple[str, int]) \
            -> tuple[Optional[socket.socket], Optional[tuple[str, int]]]:
        """Retrieves, accepts, and returns a pending connection socket from the OS connection backlog if there is one.
        The connection socket can either be connected to an arbitrary endpoint or to a pre-defined remote peer (remote
        address). If the remote address is not pre-defined, returns any connection socket that does not belong to
        another (registered) endpoint socket, otherwise stores them in the shared data structures. The same is done the
        other way around with the pending connection queue.

        :param l_addr: Address of listen socket.
        :param remote_addr: Address of remote endpoint to be connected to.
        :return: Tuple of a connection socket and the address of the remote peer.
        """
        cls._cls_logger.debug(f"Trying to accept socket for {l_addr, remote_addr}...")
        with cls._lock:
            l_sock, l_sock_lock = cls._listen_socks[l_addr]
            acc_r_socks, acc_r_lock = cls._acc_r_socks[l_addr]
            acc_p_socks = cls._acc_p_socks[l_addr]

        with l_sock_lock:
            if not select.select([l_sock], [], [], 0)[0]:
                raise RuntimeError(f"Could not open connection socket for {l_addr, remote_addr}!")
            a_sock, a_addr = l_sock.accept()
        a_addr = _convert_addr_to_name(a_addr)

        # 1. If it is the predefined remote peer, uses it for Endpoint.
        if remote_addr is not None:
            for _, _, _, _, r_addr in socket.getaddrinfo(*remote_addr, type=socket.SOCK_STREAM):
                r_addr = _convert_addr_to_name(r_addr)
                if r_addr == a_addr:
                    return a_sock, a_addr

        # 2. If it is an already registered remote peer, puts it into cache.
        with cls._lock, acc_r_lock:
            if a_addr in cls._reg_r_addrs:
                cls._cls_logger.debug(f"Storing accept socket {a_sock, a_addr} into registered connection cache...")
                _close_socket(acc_r_socks.pop(a_addr, None))
                acc_r_socks[a_addr] = a_sock
                return None, None

        # 3. If the predefined remote peer is undefined, uses it for Endpoint.
        if remote_addr is None:
            return a_sock, a_addr

        # Any other connection is stored in the pending connection queue.
        try:
            cls._cls_logger.debug(f"Storing accept socket {a_sock, a_addr} into pending connection queue...")
            acc_p_socks.put_nowait((a_sock, a_addr))
        except queue.Full:
            _close_socket(a_sock)
        return None, None

    @classmethod
    def _get_c_socket(cls, addr: tuple[str, int], remote_addr: tuple[str, int]) -> socket.socket:
        """Creates and returns a connection socket to a given remote address, that is also bound to a specific address,
        if given.
        
        :param addr: Local address to bind endpoint to.
        :param remote_addr: Address of remote endpoint to be connected to.
        :raises RuntimeError: If none of the addresses succeed to create a working socket.
        :return: The connection socket.
        """
        cls._cls_logger.debug(f"Trying to open connection socket for {addr, remote_addr}...")
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
                    sock.connect(r_addr)
                except OSError:
                    _close_socket(sock)
                    continue
                return sock
        raise RuntimeError(f"Could not open connection socket for {addr, remote_addr}!")


class StreamEndpoint:
    """One of a pair of endpoints that is able to communicate with one another over a persistent stateless stream over
    BSD sockets. Allows the transmission of generic objects in both synchronous and asynchronous fashion. Supports SSL
    and LZ4 compression for the stream. Thread-safe for both access to the same endpoint and using multiple threads
    using endpoints set to the same address.
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
    _ready: bool

    def __init__(self, name: str, addr: tuple[str, int], remote_addr: tuple[str, int] = None,
                 acceptor: bool = True, send_b_size: int = 65536, recv_b_size: int = 65536,
                 compression: bool = False, marshal_f: Callable[[object], bytes] = pickle.dumps,
                 unmarshal_f: Callable[[bytes], object] = pickle.loads,
                 multithreading: bool = False, buffer_size: int = 1024):
        """Creates a new endpoint.

        :param name: Name of endpoint for logging purposes.
        :param addr: Address of endpoint.
        :param remote_addr: Address of remote endpoint to be connected to. Optional in acceptor mode.
        :param acceptor: Determines whether the endpoint accepts or initiates connections to/from other endpoints.
        :param send_b_size: Underlying send buffer size of socket.
        :param recv_b_size: Underlying receive buffer size of socket.
        :param compression: Enables lz4 stream compression for bandwidth optimization.
        :param marshal_f: Marshal function to serialize objects to send into bytes.
        :param unmarshal_f: Unmarshal function to deserialize received bytes into objects.
        :param multithreading: Enables transparent multithreading for speedup.
        :param buffer_size: Size of shared buffer in multithreading mode.
        """
        self._logger = logging.getLogger(name)
        self._logger.info(f"Initializing endpoint {addr, remote_addr}...")

        self._endpoint_socket = EndpointSocket(name, addr, remote_addr, acceptor, send_b_size, recv_b_size)

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
        self._ready = False
        self._logger.info(f"Endpoint {addr, remote_addr} initialized.")

    def start(self):
        """Starts the endpoint, either in threaded fashion or as part of the main thread. By doing so, the two endpoints
        are connected and the datastream is opened. This method is blocking until a connection is established, but only
        in single-threaded mode.

        :raises RuntimeError: If endpoint has already been started.
        """
        self._logger.info("Starting endpoint...")
        if self._started:
            raise RuntimeError(f"Endpoint has already been started!")
        self._started = True
        if not self._multithreading:
            self._endpoint_socket.open()

        if self._multithreading:
            self._logger.info("Multithreading detected...")
            self._logger.info("\t\tstarting endpoint socket starter thread...")
            threading.Thread(target=self._create_socket_starter, daemon=True).start()
            self._logger.info("\t\tstarting endpoint sender/receiver threads...")
            self._send_thread = threading.Thread(target=self._create_sender, daemon=True)
            self._recv_thread = threading.Thread(target=self._create_receiver, daemon=True)
            self._send_thread.start()
            self._recv_thread.start()
        self._logger.info("Endpoint started.")

    def stop(self, shutdown=False, timeout=10):
        """Stops the endpoint and closes the stream, cleaning up underlying structures. If multithreading is enabled,
        waits for both endpoint threads to stop before finishing. Note this does not guarantee the sending and
        receiving of all objects still pending --- they may still be in internal buffers and will be processed if the
        endpoint is opened again, or get discarded by the underlying socket.

        :param shutdown: If set, also cleans up underlying datastructures of the socket communication.
        :param timeout: If shutdown not set, allows the sender thread to process remaining messages until timeout.
        :raises RuntimeError: If endpoint has not been started.
        """
        self._logger.info("Stopping endpoint...")
        if not self._started:
            raise RuntimeError(f"Endpoint has not been started!")

        if not shutdown and self._multithreading:
            start = time()
            while not self._send_buffer.empty() and time() - start > timeout:
                sleep(1)
        self._started = False
        self._ready = False
        self._endpoint_socket.close(shutdown)

        if self._multithreading:
            self._logger.info("Multithreading detected, waiting for endpoint sender/receiver threads to stop...")
            self._send_thread.join()
            self._recv_thread.join()
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
        :raises RuntimeError: If endpoint has not been started or has been stopped.
        :raises TimeoutError: If timeout set and triggered.
        """
        self._logger.debug("Receiving object...")
        if not self._started:
            raise RuntimeError("Endpoint has not been started!")

        p_obj = None
        if self._multithreading:
            self._logger.debug(
                f"Multithreading detected, retrieving object from buffer (size={self._recv_buffer.qsize()})...")
            while self._started:
                try:
                    if timeout is not None:
                        p_obj = self._recv_buffer.get(timeout=timeout)
                    else:
                        p_obj = self._recv_buffer.get(timeout=10)
                    break
                except queue.Empty:
                    if timeout is not None:
                        raise TimeoutError
                    continue
        else:
            p_obj = self._endpoint_socket.recv(timeout)

        if p_obj is None:
            raise RuntimeError("Endpoint has not been started!")
        self._logger.debug(f"Pickled data received of size {len(p_obj)}.")
        return self._unmarshal_f(p_obj)

    def _create_socket_starter(self):
        """Starts and connects the endpoint socket to the remote endpoint, before setting the semaphore to allow async
        sender and receiver threads to fully start.
        """
        self._endpoint_socket.open()
        self._ready = True

    def _create_sender(self):
        """Starts the loop to send objects over the socket retrieved from the sending buffer.
        """
        self._logger.info(f"AsyncSender: Starting...")
        while not self._ready:
            sleep(1)
        self._logger.info(f"AsyncSender: Starting to send objects in asynchronous mode...")

        while self._started:
            try:
                p_obj = self._send_buffer.get(timeout=10)
                self._logger.debug(
                    f"AsyncSender: Retrieved and sending object (size: {len(p_obj)}) from buffer "
                    f"(length: {self._send_buffer.qsize()})")
                self._endpoint_socket.send(p_obj)
            except queue.Empty:
                self._logger.debug(f"AsyncSender: Timeout triggered: Buffer empty. Retrying...")
        self._logger.info(f"AsyncSender: Stopping...")

    def _create_receiver(self):
        """Starts the loop to receive objects over the socket and store them in the receiving buffer.
        """
        self._logger.info(f"AsyncReceiver: Starting...")
        while not self._ready:
            sleep(1)
        self._logger.info(f"AsyncReceiver: Starting to receive objects in asynchronous mode...")

        while self._started:
            try:
                p_obj = self._endpoint_socket.recv()
                if p_obj is None:
                    continue
                self._logger.debug(
                    f"AsyncReceiver: Storing received object (size: {len(p_obj)}) in buffer "
                    f"(length: {self._recv_buffer.qsize()})...")
                self._recv_buffer.put(p_obj, timeout=10)
            except queue.Full:
                self._logger.warning(f"AsyncReceiver: Timeout triggered: Buffer full. Discarding object...")
        self._logger.info(f"AsyncReceiver: Stopping...")

    def __iter__(self):
        while self._started:
            try:
                yield self.receive()
            except RuntimeError:
                break

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop(shutdown=True)

    def __del__(self):
        if self._started:
            self.stop(shutdown=True)


def _convert_addr_to_name(addr: tuple) -> tuple[str, int]:
    """Translates a socket address, which is either a 2-tuple (ipv4) or a 4-tuple (ipv6) into a 2-tuple (host, port).
    Tries to resolve the host to its (DNS) hostname, otherwise keeps the numeric representation. Ports/Services are
    always kept numeric.

    :param addr: Address (ipv4/6) to convert.
    :return: Address tuple.
    """
    return socket.getnameinfo(addr, socket.NI_NUMERICSERV)[0], \
        int(socket.getnameinfo(addr, socket.NI_NUMERICSERV)[1])


# noinspection PyTypeChecker
def _send_payload(sock: socket.socket, payload: bytes):
    """Sends a payload over a socket, performing simple marshalling (size is sent first, then the bytes of the object).
    Blocking (if passed socket not configured otherwise).

    :param sock: Sockets to send payload over
    :param payload: Payload to send.
    """
    payload_size = len(payload)
    p_payload_size = bytes(ctypes.c_uint32(payload_size))
    _send_n_data(sock, p_payload_size, 4)
    _send_n_data(sock, payload, payload_size)


def _send_n_data(sock: socket.socket, data: bytes, size: int):
    """Sends a number of bytes over a socket.

    :param sock: Sockets to send bytes over.
    :param data: Bytes to send.
    :param size: Number of bytes to send.
    """
    sent_bytes = 0
    while sent_bytes < size:
        n_sent_bytes = sock.send(data[sent_bytes:])
        if n_sent_bytes == 0:
            raise RuntimeError("Connection broken!")
        sent_bytes += n_sent_bytes


def _recv_payload(sock: socket.socket) -> bytes:
    """Receives a payload over a socket, performing simple marshalling (size is received first, then the bytes of the
    object). Blocking (if passed socket not configured otherwise).

    :param sock: Socket to received payload over.
    :return: Received Payload.
    """
    p_payload_size = _recv_n_data(sock, 4, 1)
    payload_size = int.from_bytes(p_payload_size, byteorder=sys.byteorder)
    return _recv_n_data(sock, payload_size)


def _recv_n_data(sock: socket.socket, size: int, buff_size: int = 4096) -> bytes:
    """Receives a number of bytes over a socket.

    :param sock: Socket to receive bytes over.
    :param size: Number of bytes to receive.
    :param buff_size: Maximum number of bytes to receive from socket per receive iteration.
    :return: Received n bytes.
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

    :param sock: Socket to close.
    """
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()
