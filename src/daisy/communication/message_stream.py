# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""An efficient, persistent, and stateless communications stream between two
endpoints over BSD sockets. Supports SSL (soon) and LZ4 compression.

Author: Fabian Hofmann
Modified: 02.04.24
"""
# TODO Future Work: SSL https://docs.python.org/3/library/ssl.html
# TODO Future Work: Defining granularity of logging in inits
# TODO Future Work: Endpoint socket spam may result in re-use of wrong "old" connection
#   1. acceptor accepts connection of generic initiator, registers its address
#   2. initiator crashes/shut downs, acceptor keeps trying to re-establish
#   3. another initiator from same host (or nat) gets recycled address IP and connects
#   4. acceptor accepts this "wrong" endpoint, believing it is still the old one
#   - Should be fixed from the ground up (possible through two-way, cert-based identity)
# TODO Future Work: Allow finite communication sessions (keep-alive vs graceful close)
# FIXME: Race Conditions:
#   - Diff between select and check_r_sock
#   - Double Shutdowns of Socks (Unreg is called twice)
# FIXME: Rapid single-use endpoints clog system, especially ep-server (+multithreading)
# FIXME: Cleanup Thread in ep-server clogs up due to threading spam + sleeps + locks
# TODO Future Work: Async Stop of Endpoints (both for endpoints and ep-servers)

import ctypes
import logging
import pickle
import queue
import select
import socket
import sys
import threading
from time import sleep, time
from typing import Callable, Iterable, Optional, Self

from lz4.frame import compress, decompress


class EndpointSocket:
    """A bundle of up to two sockets, that is used to communicate with another
    endpoint over a persistent TCP connection in synchronous manner. Supports
    authentication and encryption over SSL, and stream compression using LZ4.
    Thread-safe for both access to the same endpoint socket and using multiple
    threads using endpoint sockets set to the same address (this is organized through
    an array of class variables, see below for more info).

    :cvar _listen_socks: Active listen sockets, along with a respective lock to access
    each safely.
    :cvar _acc_r_socks: Pending registered connection cache for each listen socket.
    :cvar _acc_p_socks: Pending unregistered connection queue for each listen socket.
    :cvar _reg_r_addrs: Registered remote addresses.
    :cvar _addr_map: Mapping between registered remote addresses and their aliases.
    :cvar _act_l_counts: Active thread counter for each listen socket. Socket closes
    if counter reaches zero.
    :cvar _lock: General purpose lock to ensure safe access to class variables.
    :cvar _cls_logger: General purpose logger for class methods.
    """

    _listen_socks: dict[tuple[str, int], tuple[socket.socket, threading.Lock]] = {}
    _acc_r_socks: dict[
        tuple[str, int], tuple[dict[tuple[str, int], socket.socket], threading.Lock]
    ] = {}
    _acc_p_socks: dict[
        tuple[str, int], queue.Queue[tuple[socket.socket, tuple[str, int]]]
    ] = {}
    _reg_r_addrs: set[tuple[str, int]] = set()
    _addr_map: dict[tuple[str, int], set[tuple[str, int]]] = {}
    _act_l_counts: dict[tuple[str, int], int] = {}
    _lock = threading.Lock()
    _cls_logger: logging.Logger = logging.getLogger("EndpointSocketCLS")

    _logger: logging.Logger

    _addr: tuple[str, int]
    _remote_addr: Optional[tuple[str, int]]
    _acceptor: bool

    _send_b_size: int
    _recv_b_size: int
    _sock: Optional[socket.socket]
    _sock_lock: threading.Lock

    _conn_rdy: threading.Event
    _opened: bool

    def __init__(
        self,
        name: str,
        addr: tuple[str, int] = None,
        remote_addr: tuple[str, int] = None,
        acceptor: bool = True,
        send_b_size: int = 65536,
        recv_b_size: int = 65536,
    ):
        """Creates a new endpoint socket. Implementation note: A pre-defined remote
        address is not a guarantee that this endpoint will successfully be allowed to
        initialize for this remote address --- for example if another endpoint sock
        with the same remote address (be it generic or pre-defined) has already been
        registered, then the current one will throw an error.

        :param name: Name of endpoint for logging purposes.
        :param addr: Address of endpoint. Mandatory in acceptor mode (acceptor set to
        True), for initiators this fixes the address the endpoint is bound to.
        :param remote_addr: Address of remote endpoint to be connected to. Mandatory in
        initiator mode (acceptor set to false), for acceptors this fixes the remote
        endpoint that is allowed to be connected to this endpoint.
        :param acceptor: Determines whether the endpoint accepts or initiates
        connections to/from other endpoints.
        :param send_b_size: Underlying send buffer size of socket.
        :param recv_b_size: Underlying receive buffer size of socket.
        :raises ValueError: If the remote address is already taken for the acceptor,
        or if the address/remote address is not provided for acceptor/initiator.
        """
        self._logger = logging.getLogger(name + "-Socket")
        self._logger.info(f"Initializing endpoint socket {addr, remote_addr}...")

        self._addr = addr if addr is not None else ("0.0.0.0", 0)
        self._remote_addr = remote_addr
        self._acceptor = acceptor
        if acceptor:
            if addr is None:
                raise ValueError("Accepting endpoint socket requires an address!")
            elif remote_addr is not None:
                self._reg_remote(remote_addr)
                self._fix_rp_acc_socks(addr, remote_addr)
        elif remote_addr is None:
            raise ValueError("Initiating endpoint socket requires a remote address!")

        self._send_b_size = send_b_size
        self._recv_b_size = recv_b_size
        self._sock = None
        self._sock_lock = threading.Lock()

        self._conn_rdy = threading.Event()
        self._opened = False
        self._logger.info(f"Endpoint socket {addr, remote_addr} initialized.")

    def open(self):
        """Opens the endpoint socket along with its underlying socket(s) and its
        connection to a/the remote endpoint socket. Blocking until the connection is
        established.
        """
        self._logger.info("Opening endpoint socket...")
        self._opened = True
        if self._acceptor:
            self._open_l_socket(self._addr)
        self._conn_rdy.set()
        self._connect()
        self._logger.info("Endpoint socket opened.")

    def close(self, shutdown: bool = False):
        """Closes the endpoint socket, cleaning up any underlying datastructures if
        acceptor. If already closed, allows the cleanup of just the datastructures
        incase a shutdown is requested.
        """
        if (
            not self._opened
            and shutdown
            and self._acceptor
            and self._remote_addr is not None
        ):
            self._unreg_remote(self._addr, self._remote_addr)
            return

        self._logger.info("Closing endpoint socket...")
        self._opened = False
        with self._sock_lock:
            _close_socket(self._sock)
            self._sock = None
            if self._acceptor:
                if shutdown and self._remote_addr is not None:
                    self._unreg_remote(self._addr, self._remote_addr)
                self._close_l_socket(self._addr)
        self._logger.info("Endpoint socket closed.")

    def send(self, p_data: bytes):
        """Sends the given bytes of a single object over the connection, performing
        simple marshalling (size is sent first, then the bytes of the object).
        Fault-tolerant for breakdowns and resets in the connection. Blocking.

        :param p_data: Bytes to send.
        """
        while self._opened:
            try:
                self._sock_lock.acquire()
                if _check_w_socket(self._sock):
                    _send_payload(self._sock, p_data)
                    self._sock_lock.release()
                    return
                self._sock_lock.release()
                sleep(1)
            except (OSError, ValueError, RuntimeError) as e:
                self._logger.warning(
                    f"{e.__class__.__name__}({e}) "
                    "while trying to send data. Retrying..."
                )
                # unlock is done in connect()
                self._connect()

    def recv(self, timeout: int = None) -> Optional[bytes]:
        """Receives the bytes of a single object sent over the connection, performing
        simple marshalling (size is received first, then the bytes of the object).
        Fault-tolerant for breakdowns and resets in the connection. Blocking in
        default-mode if timeout not set.

        :param timeout: Timeout (seconds) to receive an object to return.
        :return: Received bytes or None of end point socket has been closed.
        :raises TimeoutError: If timeout set and triggered.
        """
        while self._opened:
            try:
                self._sock_lock.acquire()
                if _check_r_socket(self._sock, timeout=timeout):
                    p_data = _recv_payload(self._sock)
                    self._sock_lock.release()
                    return p_data
                self._sock_lock.release()
                sleep(1)
            except (OSError, ValueError, RuntimeError) as e:
                if timeout is not None and type(e) is TimeoutError:
                    raise e
                self._logger.warning(
                    f"{e.__class__.__name__}({e}) "
                    "while trying to receive data. Retrying..."
                )
                # unlock is done in connect()
                self._connect()
        return None

    def poll(
        self, lazy: bool = False
    ) -> tuple[list[bool], tuple[tuple[str, int], tuple[str, int]]]:
        """Polls the state of various state and addresses of the endpoint socket:
            * 0,0: Existence of socket (true if connected).
            * 0,1: Whether there is something to read on the underlying socket.
            * 0,2: Whether one is able to write on the underlying socket.
            + 1,0: Address of endpoint socket, else None
            + 1,1: Address of remote endpoint socket, else None.
        Note this does not necessarily guarantee that the underlying socket is
        actually connected and available for reading/writing; e.g. the connection
        could have broken down since then and is currently being re-established.

        :param lazy: Whether to lazily skip the actual state of the underlying socket
        and just check for connectivity.
        :return: Tuple of boolean states (connectivity, readability, writability) and
        address-pair of endpoint socket.
        """
        states = [self._sock is not None] + [False] * 2
        if not lazy:
            with self._sock_lock:
                states[0] = self._sock is not None
                if self._sock is not None:
                    states[1] = len(select.select([self._sock], [], [], 0)[0]) != 0
                    states[2] = len(select.select([], [self._sock], [], 0)[1]) != 0
        return states, (self._addr, self._remote_addr)

    def _connect(self):
        """TODO(Re-)Establishes a connection to a/the remote endpoint socket,
        first performing any necessary cleanup of the underlying socket,
        before opening it again and trying to connect/accept a remote endpoint
        socket. Fault-tolerant for breakdowns and resets in the connection. Blocking.
        """
        if not self._conn_rdy.is_set():
            self._sock_lock.release()
            self._conn_rdy.wait()
            return
        self._conn_rdy.clear()

        while self._opened:
            try:
                self._logger.info(
                    f"Trying to (re-)establish connection "
                    f"{self._addr, self._remote_addr}..."
                )
                self._setup()
                self._logger.info(
                    f"Connection {self._addr, self._remote_addr} (re-)established."
                )
                break
            except (OSError, ValueError, AttributeError, RuntimeError) as e:
                self._logger.info(
                    f"{e.__class__.__name__}({e}) while trying to (re-)establish "
                    f"connection {self._addr, self._remote_addr}. Retrying..."
                )
                self._sock_lock.release()
                sleep(10)
                self._sock_lock.acquire()
        self._conn_rdy.set()
        self._sock_lock.release()

    def _setup(self):
        """TODO

        :return:
        """
        _close_socket(self._sock)
        self._sock = None
        if self._acceptor:
            remote_addr = self._remote_addr
            while self._opened and self._sock is None:
                self._sock, remote_addr = self._get_a_socket(
                    self._addr, self._remote_addr
                )
            self._remote_addr = remote_addr
        else:
            self._sock, self._addr = self._get_c_socket(self._addr, self._remote_addr)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self._send_b_size)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._recv_b_size)

    @classmethod
    def _reg_remote(cls, remote_addr: tuple[str, int]):
        """Registers a remote address into the class datastructures, notifying other
        endpoints of its existence. Tries to both resolve the address and finds its
        fully qualified hostname to reserve all its aliases. If only a single alias
        is already registered, aborts the whole registration process and registers
        none of the aliases.

        :param remote_addr: Remote address to register.
        :raises ValueError: If remote address is already registered (possibly by
        another caller).
        """
        cls._cls_logger.info(f"Registering remote address ({remote_addr})...")
        addr_mapping = set()
        for _, _, _, _, addr in socket.getaddrinfo(
            *remote_addr, type=socket.SOCK_STREAM
        ):
            addr = _convert_addr_to_name(addr)
            addr_mapping.add(addr)

        cls._cls_logger.debug(
            f"Registering aliases of remote address ({remote_addr}): {addr_mapping}..."
        )
        with cls._lock:
            if remote_addr in cls._addr_map:
                raise ValueError(
                    f"Remote address ({remote_addr}) is already registered!"
                )
            for addr in addr_mapping:
                if addr in cls._reg_r_addrs:
                    raise ValueError(
                        f"Remote address ({addr}) (resolved from {remote_addr})"
                        " is already registered!"
                    )

            cls._addr_map[remote_addr] = addr_mapping
            for addr in addr_mapping:
                cls._reg_r_addrs.add(addr)
        cls._cls_logger.info(f"Remote address ({remote_addr}) registered.")

    @classmethod
    def _fix_rp_acc_socks(cls, addr: tuple[str, int], remote_addr: tuple[str, int]):
        """After a remote address has been registered, cycles the waiting connections
        of that remote address to the registered socket dictionary from the pending
        socket queue. Necessary, as new endpoint sockets may be created while the
        listening socket is already opened and connections are already getting accepted.

        Note this method merely helps in speeding up the sorting of connections to
        the correct endpoint socket and operates in best-effort manner, as the whole
        method is not considered a critical section, i.e. during the cycling of
        connections, the registered connection could be accepted by another thread
        and be put into the pending connection cache, where it will be dequeued by
        another thread --- which will result in a ValueError since the remote peer is
        obviously registered. However, no much of an issue, as the error handling is
        done in the background during the handling of existing connections.

        :param addr: Address of endpoint.
        :param remote_addr: Remote address that was registered.
        """
        cls._cls_logger.info(
            f"Fixing registered and pending connection caches for address pair "
            f"{addr, remote_addr}..."
        )
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
            "Registered and pending connection caches for "
            f"address pair {addr, remote_addr} fixed."
        )

    @classmethod
    def _unreg_remote(cls, addr: tuple[str, int], remote_addr: tuple[str, int]):
        """Unregisters a remote address from the class datastructures. Uses the
        existing mappings from the original resolution. Also cycles any pending
        connections of that remote address from the registered socket dictionary to
        the pending socket queue so the connection may be accepted by any other
        (generic or pre-defined) endpoint socket listening on the same address.

        :param remote_addr: Remote address that was registered.
        """
        cls._cls_logger.info(
            f"Unregistering remote address pair {addr, remote_addr}..."
        )
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
        """Opens the socket listening to a given address, iff there are no further
        endpoint sockets listening on the same socket as well.

        :param addr: Address of listen socket to open.
        """
        cls._get_l_socket(addr, new_endpoint=True)

    @classmethod
    def _get_l_socket(
        cls, addr: tuple[str, int], new_endpoint: bool = False
    ) -> tuple[tuple[str, int], socket.socket, threading.Lock]:
        """Gets the socket listening to a given address. If this socket does not
        exist already, creates it and with it all accompanying datastructures.
        Supports address resolution.

        :param addr: Address of listen socket.
        :return: A tupel consisting of the address, the socket, and a lock to be used
        for accessing the socket.
        :raises RuntimeError: If none of the addresses/aliases succeed to create a
        working socket.
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
            cls._cls_logger.debug(
                f"Listening socket {l_addr, l_sock, l_sock_lock} for {addr} retrieved"
            )
            return l_addr, l_sock, l_sock_lock
        raise RuntimeError(f"Could not open listen socket for {addr}!")

    @classmethod
    def _close_l_socket(cls, addr: tuple[str, int]):
        """Closes the socket listening to a given address, iff there are no further
        endpoint sockets listening on the same socket as well.

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
    def _get_a_socket(
        cls, addr: tuple[str, int], remote_addr: tuple[str, int] = None
    ) -> tuple[Optional[socket.socket], Optional[tuple[str, int]]]:
        """Gets an accepted connection socket assigned to a socket of a given
        address. The connection socket can either be connected to an arbitrary
        endpoint or to a pre-defined remote peer (remote address). If remote address
        is not pre-defined, also registers the remote peer's address. As the
        underlying datastructures are shared between all endpoint sockets (of a
        process), a connection socket can either be retrieved from them or directly
        from the listen socket.

        :param addr: Address of listen socket.
        :param remote_addr: Address of remote endpoint to be connected to.
        :return: Tuple of the connection socket and the address of the remote peer.

        :raises RuntimeError: If none of the addresses/aliases of the listen socket
        succeed to get a working socket.
        """
        cls._cls_logger.debug(
            f"Trying to retrieve accept socket for {addr, remote_addr}..."
        )
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
    def _get_r_acc_sock(
        cls, l_addr: tuple[str, int], remote_addr: tuple[str, int]
    ) -> Optional[socket.socket]:
        """Retrieves and returns a registered (accepted) connection socket assigned
        to a socket of a given address, if it exists in the (shared) active
        registered connection cache.

        :param l_addr: Address of listen socket.
        :param remote_addr: Address of remote endpoint to be connected to.
        :return: Tuple of the registered connection socket and the address of the
        remote peer.
        """
        cls._cls_logger.debug(
            f"Trying to retrieve accept socket for {l_addr, remote_addr} "
            "from registered connection cache..."
        )
        with cls._lock:
            acc_r_socks, acc_r_lock = cls._acc_r_socks[l_addr]
        for _, _, _, _, addr in socket.getaddrinfo(
            *remote_addr, type=socket.SOCK_STREAM
        ):
            addr = _convert_addr_to_name(addr)
            with acc_r_lock:
                a_sock = acc_r_socks.pop(addr, None)
            if a_sock is not None:
                return a_sock
        return None

    @classmethod
    def _get_p_acc_sock(
        cls, l_addr: tuple[str, int]
    ) -> tuple[Optional[socket.socket], Optional[tuple[str, int]]]:
        """Retrieves and returns a pending, not registered (accepted) connection
        socket assigned to a socket of a given address, if there is one in the
        pending connection queue.

        :param l_addr: Address of listen socket.
        :return: Tuple of a connection socket and the address of the remote peer.
        """
        cls._cls_logger.debug(
            f"Trying to retrieve accept socket for {l_addr} "
            "from pending connection queue..."
        )
        with cls._lock:
            acc_p_socks = cls._acc_p_socks[l_addr]
        try:
            return acc_p_socks.get_nowait()
        except queue.Empty:
            return None, None

    @classmethod
    def _get_n_acc_sock(
        cls, l_addr: tuple[str, int], remote_addr: tuple[str, int]
    ) -> tuple[Optional[socket.socket], Optional[tuple[str, int]]]:
        """Retrieves, accepts, and returns a pending connection socket from the OS
        connection backlog if there is one. The connection socket can either be
        connected to an arbitrary endpoint or to a pre-defined remote peer (remote
        address). If the remote address is not pre-defined, returns any connection
        socket that does not belong to another (registered) endpoint socket,
        otherwise stores them in the shared data structures. The same is done the
        other way around with the pending connection queue.

        :param l_addr: Address of listen socket.
        :param remote_addr: Address of remote endpoint to be connected to.
        :return: Tuple of a connection socket and the address of the remote peer.
        :raises RuntimeError: If there are no new connections are in the OS
        connection backlog.
        """
        cls._cls_logger.debug(f"Trying to accept socket for {l_addr, remote_addr}...")
        with cls._lock:
            l_sock, l_sock_lock = cls._listen_socks[l_addr]
            acc_r_socks, acc_r_lock = cls._acc_r_socks[l_addr]
            acc_p_socks = cls._acc_p_socks[l_addr]

        with l_sock_lock:
            if not select.select([l_sock], [], [], 0)[0]:
                raise RuntimeError(
                    f"Could not open connection socket for {l_addr, remote_addr}!"
                )
            a_sock, a_addr = l_sock.accept()
        a_addr = _convert_addr_to_name(a_addr)

        # 1. If it is the predefined remote peer, uses it for Endpoint.
        if remote_addr is not None:
            for _, _, _, _, r_addr in socket.getaddrinfo(
                *remote_addr, type=socket.SOCK_STREAM
            ):
                r_addr = _convert_addr_to_name(r_addr)
                if r_addr == a_addr:
                    return a_sock, a_addr

        # 2. If it is an already registered remote peer, puts it into cache.
        with cls._lock, acc_r_lock:
            if a_addr in cls._reg_r_addrs:
                cls._cls_logger.debug(
                    f"Storing accept socket {a_sock, a_addr} "
                    "into registered connection cache..."
                )
                _close_socket(acc_r_socks.pop(a_addr, None))
                acc_r_socks[a_addr] = a_sock
                return None, None

        # 3. If the predefined remote peer is undefined, uses it for Endpoint.
        if remote_addr is None:
            return a_sock, a_addr

        # Any other connection is stored in the pending connection queue.
        try:
            cls._cls_logger.debug(
                f"Storing accept socket {a_sock, a_addr} "
                "into pending connection queue..."
            )
            acc_p_socks.put_nowait((a_sock, a_addr))
        except queue.Full:
            _close_socket(a_sock)
        return None, None

    @classmethod
    def _get_c_socket(
        cls, addr: tuple[str, int], remote_addr: tuple[str, int]
    ) -> tuple[Optional[socket.socket], Optional[tuple[str, int]]]:
        """Creates and returns a connection socket to a given remote address,
        that might be bound to a specific address, if given. Non-Blocking (with
        timeout) during connection attempts.

        :param addr: Local address to bind endpoint to. If none provided, OS chooses
        an address.
        :param remote_addr: Address of remote endpoint to be connected to.
        :return: Tuple of the connection socket and the address of the socket.
        :raises RuntimeError: If no connection can be established.
        """
        cls._cls_logger.debug(
            f"Trying to open connection socket for {addr, remote_addr}..."
        )
        for res in socket.getaddrinfo(*addr, type=socket.SOCK_STREAM):
            s_af, s_t, s_p, _, s_addr = res
            r_res_list = socket.getaddrinfo(
                *remote_addr, family=s_af.value, type=s_t.value, proto=s_p
            )
            for r_res in r_res_list:
                r_af, r_t, r_p, _, r_addr = r_res
                sock = None
                try:
                    sock = socket.socket(r_af, r_t, r_p)
                    sock.settimeout(10)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind(s_addr)
                    sock.connect(r_addr)
                    sock.settimeout(None)
                except OSError:
                    _close_socket(sock)
                    continue
                return sock, _convert_addr_to_name(sock.getsockname())
        raise RuntimeError(f"Could not open connection socket for {addr, remote_addr}!")


class StreamEndpoint:
    """One of a pair of endpoints that is able to communicate with one another over a
    persistent stateless stream over BSD sockets. Allows the transmission of generic
    objects in both synchronous and asynchronous fashion. Supports SSL and LZ4
    compression for the stream. Thread-safe for both access to the same endpoint and
    using multiple threads using endpoints set to the same address.
    """

    _logger: logging.Logger

    _endpoint_socket: EndpointSocket
    _marshal_f: Callable[[object], bytes]
    _unmarshal_f: Callable[[bytes], object]

    _multithreading: bool
    _sender: threading.Thread
    _receiver: threading.Thread
    _send_buffer: queue.Queue[bytes]
    _recv_buffer: queue.Queue[bytes]

    _started: bool
    _ready: threading.Event
    _shutdown: bool

    def __init__(
        self,
        name: str = "StreamEndpoint",
        addr: tuple[str, int] = None,
        remote_addr: tuple[str, int] = None,
        acceptor: bool = True,
        send_b_size: int = 65536,
        recv_b_size: int = 65536,
        compression: bool = False,
        marshal_f: Callable[[object], bytes] = pickle.dumps,
        unmarshal_f: Callable[[bytes], object] = pickle.loads,
        multithreading: bool = False,
        buffer_size: int = 1024,
    ):
        """Creates a new endpoint.

        :param name: Name of endpoint for logging purposes.
        :param addr: Address of endpoint.
        :param remote_addr: Address of remote endpoint to be connected to. Optional in
        acceptor mode.
        :param acceptor: Determines whether the endpoint accepts or initiates
        connections to/from other endpoints.
        :param send_b_size: Underlying send buffer size of socket.
        :param recv_b_size: Underlying receive buffer size of socket.
        :param compression: Enables lz4 stream compression for bandwidth optimization.
        :param marshal_f: Marshal function to serialize objects to send into bytes.
        :param unmarshal_f: Unmarshal function to deserialize received bytes into
        objects.
        :param multithreading: Enables transparent multithreading (i.e. asynchronous
        object processing) for speedup.
        :param buffer_size: Size of shared buffer in multithreading mode.
        """
        self._logger = logging.getLogger(name)
        self._logger.info(f"Initializing endpoint {addr, remote_addr}...")

        self._endpoint_socket = EndpointSocket(
            name=name,
            addr=addr,
            remote_addr=remote_addr,
            acceptor=acceptor,
            send_b_size=send_b_size,
            recv_b_size=recv_b_size,
        )

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
        self._ready = threading.Event()
        self._shutdown = False
        self._logger.info(f"Endpoint {addr, remote_addr} initialized.")

    def start(self, blocking=True) -> threading.Event:
        """Starts the endpoint, either in threaded fashion or as part of the main
        thread. By doing so, the two endpoints are connected and the datastream is
        opened. This method is blocking until a connection is established by default
        if multithreading is not enabled or the respective flag is not set. If either
        is the case, the caller can check the readiness of the connection via the
        returned event object. Note that in multithreading mode, objects can already
        be sent/received, however they will only be stored in internal buffers until
        the connection is established!

        :param blocking: Whether to wait for a connection to be established in
        non-multithreading (sync) mode.
        :return: Event object to check endpoint's readiness to send/receive. Always
        true if start() was called blocking.
        :raises RuntimeError: If endpoint has already been started or shut down.
        """
        self._logger.info("Starting endpoint...")
        if self._shutdown:
            raise RuntimeError("Endpoint has already been shut down!")
        if self._started:
            raise RuntimeError("Endpoint has already been started!")
        self._started = True

        if not blocking or self._multithreading:
            self._logger.info("Starting endpoint socket starter thread...")
            threading.Thread(target=self._create_socket_starter, daemon=True).start()
        else:
            self._create_socket_starter()

        if self._multithreading:
            self._logger.info(
                "Multithreading detected, starting endpoint sender/receiver threads..."
            )
            self._sender = threading.Thread(target=self._create_sender, daemon=True)
            self._receiver = threading.Thread(target=self._create_receiver, daemon=True)
            self._sender.start()
            self._receiver.start()
        self._logger.info("Endpoint started.")
        return self._ready

    def stop(self, shutdown=False, timeout=10):
        """Stops the endpoint and closes the stream, cleaning up underlying
        datastructures. If multithreading is enabled, waits for both endpoint threads
        to stop before finishing. Note this does not guarantee the sending and
        receiving of all objects still pending --- they may still be in internal
        buffers and will be processed if the endpoint is opened again,
        or get discarded by the underlying socket.

        Also note if the endpoint has not been started or has already been closed,
        a set shutdown flag still results in the full cleanup of the underlying
        datastructures.

        :param shutdown: If set, also cleans up underlying datastructures of the
        socket communication.
        :param timeout: Allows the sender thread to process remaining messages until
        timeout.
        :raises RuntimeError: If endpoint has not been started or already shut down.
        """
        self._logger.info("Stopping endpoint...")
        if not self._started:
            if not self._shutdown and shutdown:
                self._logger.warning(
                    "Shutdown on closed endpoint detected, cleaning up endpoint..."
                )
                self._endpoint_socket.close(shutdown)
                self._shutdown = True
                return
            else:
                raise RuntimeError(
                    "Endpoint has not been started or already shut down!"
                )

        if self._multithreading:
            start = time()
            while not self._send_buffer.empty() and time() - start < timeout:
                sleep(1)
        self._started = False
        self._ready.set()
        self._endpoint_socket.close(shutdown)

        if self._multithreading:
            self._logger.info(
                "Multithreading detected, waiting for "
                "endpoint sender/receiver threads to stop..."
            )
            self._sender.join()
            self._receiver.join()
        self._ready.clear()
        self._logger.info("Endpoint stopped.")

    def send(self, obj: object):
        """Generic send function that sends any object as a pickle over the
        persistent datastream. If multithreading is enabled, this function is
        non-blocking.

        :param obj: Object to send.
        :raises RuntimeError: If endpoint has not been started.
        """
        self._logger.debug("Sending object...")
        if not self._started:
            raise RuntimeError("Endpoint has not been started!")

        p_obj = self._marshal_f(obj)
        if self._multithreading:
            self._logger.debug(
                "Multithreading detected, putting object "
                f"into buffer (size={self._send_buffer.qsize()})..."
            )
            self._send_buffer.put(p_obj)
        else:
            self._endpoint_socket.send(p_obj)
        self._logger.debug(f"Pickled object sent of size {len(p_obj)}.")

    def receive(self, timeout: int = None) -> object:
        """Generic receive function that receives data as a pickle over the
        persistent datastream, unpickles it into the respective object and returns
        it. Blocking in default-mode if timeout not set.

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
                "Multithreading detected, retrieving object "
                f"from buffer (size={self._recv_buffer.qsize()})..."
            )
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

    def poll(self) -> tuple[list[bool], tuple[tuple[str, int], tuple[str, int]]]:
        """Polls the state of various stats of the endpoint (see below) and addresses
        of endpoint.
            * 0,0: Existence of underlying socket (true if connected).
            * 0,1: Whether there is something to read on the internal buffer (async)
                or underlying socket (sync).
            * 0,2: Whether one is able to write on the internal buffer (async)
                or underlying socket (sync).
            + 1,0: Address of endpoint, else None
            + 1,1: Address of remote endpoint, else None.
        Note this does not necessarily guarantee that the underlying endpoint socket
        is actually connected and available for reading/writing; not only could have
        the connection broken down since then and is being re-established,
        but in multithreading mode the content of the internal buffers might change
        over time and thus change the read/write state.

        :return: Tuple of boolean states (connectivity, readability, writability) and
        address-pair of endpoint.
        """
        states, addrs = self._endpoint_socket.poll(lazy=self._multithreading)
        if self._multithreading:
            states[1] = not self._recv_buffer.empty()
            states[2] = not self._send_buffer.full() and states[0]
        return states, addrs

    def _create_socket_starter(self):
        """Starts and connects the endpoint socket to the remote endpoint, before
        setting the semaphore to allow async sender and receiver threads to fully start.
        """
        self._endpoint_socket.open()
        self._ready.set()

    def _create_sender(self):
        """Starts the loop to send objects over the socket retrieved from the sending
        buffer.
        """
        self._logger.info("AsyncSender: Starting...")
        self._ready.wait()
        self._logger.info(
            "AsyncSender: Starting to send objects in asynchronous mode..."
        )

        while self._started:
            try:
                p_obj = self._send_buffer.get(timeout=10)
                self._logger.debug(
                    f"AsyncSender: Retrieved and sending object (size: {len(p_obj)}) "
                    f"from buffer (length: {self._send_buffer.qsize()})"
                )
                self._endpoint_socket.send(p_obj)
            except queue.Empty:
                self._logger.debug(
                    "AsyncSender: Timeout triggered: Buffer empty. Retrying..."
                )
        self._logger.info("AsyncSender: Stopping...")

    def _create_receiver(self):
        """Starts the loop to receive objects over the socket and store them in the
        receiving buffer.
        """
        self._logger.info("AsyncReceiver: Starting...")
        self._ready.wait()
        self._logger.info(
            "AsyncReceiver: Starting to receive objects in asynchronous mode..."
        )

        while self._started:
            try:
                p_obj = self._endpoint_socket.recv()
                if p_obj is None:
                    continue
                self._logger.debug(
                    f"AsyncReceiver: Storing received object (size: {len(p_obj)}) "
                    f"in buffer (length: {self._recv_buffer.qsize()})..."
                )
                self._recv_buffer.put(p_obj, timeout=10)
            except queue.Full:
                self._logger.warning(
                    "AsyncReceiver: Timeout triggered: Buffer full. "
                    "Discarding object..."
                )
        self._logger.info("AsyncReceiver: Stopping...")

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
        if not self._shutdown:
            self.stop(shutdown=True)

    @classmethod
    def create_quick_sender_ep(
        cls,
        objects: Iterable,
        remote_addr: tuple[str, int],
        name: str = "QuickSenderEndpoint",
        addr: tuple[str, int] = None,
        send_b_size: int = 65536,
        compression: bool = False,
        marshal_f: Callable[[object], bytes] = pickle.dumps,
        blocking=True,
    ):
        """Creates a (simplified) one-time endpoint to send a number of objects to a
        remote endpoint before shutting down. May be called non-blocking to handle
        endpoint in background entirely.

        :param objects: Iterable of objects to send to remote endpoint.
        :param remote_addr: Address of remote endpoint to send messages to.
        :param name: Name of endpoint for logging purposes.
        :param addr: Address of endpoint.
        :param send_b_size: Underlying send buffer size of socket.
        :param compression: Enables lz4 stream compression for bandwidth optimization.
        :param marshal_f: Marshal function to serialize objects to send into byte
        :param blocking: Whether endpoint and message handling is to be done
        synchronously or asynchronously (using threads).
        """

        def quick_sender_ep():
            endpoint = StreamEndpoint(
                name=name,
                addr=addr,
                remote_addr=remote_addr,
                acceptor=False,
                send_b_size=send_b_size,
                compression=compression,
                marshal_f=marshal_f,
                multithreading=False,
            )
            endpoint.start()
            for obj in objects:
                endpoint.send(obj)
            endpoint.stop(shutdown=True)

        if blocking:
            threading.Thread(target=quick_sender_ep, daemon=True).start()
        else:
            quick_sender_ep()

    @classmethod
    def receive_latest_ep_objs(
        cls, endpoints: Iterable[Self], obj_type: type = object
    ) -> dict[Self, Optional]:
        """Endpoint helper function to receive the latest objects of a certain type
        from a number of endpoints. Note this flushes any other messages held by
        these endpoints as well, as non-blocking receives are called on them until
        their buffers are exhausted. Any messages of others types are discarded,
        as are endpoints who are not ready.

        :param endpoints: Iterable of endpoints to receive objects from.
        :param obj_type: Type of objects to receive. If none given, receives the latest
        message of any type.
        :return: Dictionary of each endpoint and their respective received object,
        None if nothing received for endpoint.
        """
        ep_objs = {}
        for endpoint in endpoints:
            ep_obj = None
            try:
                while True:
                    ep_msg = endpoint.receive(timeout=0)
                    if isinstance(ep_msg, obj_type):
                        ep_obj = ep_msg
                    else:
                        pass
            except (RuntimeError, TimeoutError):
                pass
            ep_objs[endpoint] = ep_obj
        return ep_objs

    @classmethod
    def select_eps(cls, endpoints: Iterable[Self]) -> tuple[list[Self], list[Self]]:
        """Endpoint select helper function to check a number of endpoints whether
        objects can be read from or written to them. For simplicity's sake, does not
        mirror the actual UNIX select function (supporting separate lists).

        :param endpoints: Iterable of endpoints to check for readiness.
        :return: Tuple of lists of endpoints that are read/write ready:
        """
        ep_states = [(endpoint, endpoint.poll()) for endpoint in endpoints]
        r_ready = list(map(lambda t: t[0], filter(lambda t: t[1][0][1], ep_states)))
        w_ready = list(map(lambda t: t[0], filter(lambda t: t[1][0][2], ep_states)))
        return r_ready, w_ready


class EndpointServer:
    """Helper class to manage a group of (acceptor) connection endpoints listening to
    the same address. Supports all features of the existing endpoint class, besides
    also supporting thread-safe access, polling, and management of them as a group.
    """

    _logger: logging.Logger

    _connection_handler: threading.Thread
    _connection_cleaner: threading.Thread

    _connections: dict[tuple[str, int], StreamEndpoint]
    _p_connections: queue.Queue[tuple[tuple[str, int], StreamEndpoint]]
    _n_connections: int
    _c_timeout: int
    _c_lock: threading.RLock

    _name: str
    _addr: tuple[str, int]
    _send_b_size: int
    _recv_b_size: int
    _compression: bool
    _marshal_f: Callable[[object], bytes]
    _unmarshal_f: Callable[[bytes], object]
    _multithreading: bool
    _buffer_size: int

    _started: bool

    def __init__(
        self,
        addr: tuple[str, int],
        name: str = "EndpointServer",
        c_timeout: int = None,
        send_b_size: int = 65536,
        recv_b_size: int = 65536,
        compression: bool = False,
        marshal_f: Callable[[object], bytes] = pickle.dumps,
        unmarshal_f: Callable[[bytes], object] = pickle.loads,
        multithreading: bool = False,
        buffer_size: int = 1024,
    ):
        """Creates a new endpoint server.

        :param addr: Address of endpoint server.
        :param name: Name of endpoint server for logging purposes.
        :param c_timeout: Timeout (secs) for disconnected connection endpoints when
        performing periodic cleanup.
        :param send_b_size: Underlying send buffer size of all connection sockets.
        :param recv_b_size: Underlying receive buffer size of all connection sockets.
        :param compression: Enables lz4 stream compression for bandwidth optimization.
        :param marshal_f: Marshal function to serialize objects to send into bytes.
        :param unmarshal_f: Unmarshal function to deserialize received bytes into
        objects.
        :param multithreading: Enables transparent multithreading (for individual
        endpoints) for speedup.
        :param buffer_size: Size of shared buffers, both for server and for
        connection endpoints in multithreading mode.
        """
        self._logger = logging.getLogger(name)
        self._logger.info(f"Initializing endpoint server {addr}...")

        self._connections = {}
        self._p_connections = queue.Queue(maxsize=buffer_size)
        self._n_connections = 0
        self._c_timeout = c_timeout
        self._c_lock = threading.RLock()

        self._name = name
        self._addr = addr
        self._send_b_size = send_b_size
        self._recv_b_size = recv_b_size
        self._compression = compression
        self._marshal_f = marshal_f
        self._unmarshal_f = unmarshal_f
        self._multithreading = multithreading
        self._buffer_size = buffer_size

        self._started = False
        self._logger.info(f"Endpoint server {addr} initialized.")

    def start(self):
        """Starts the endpoint server, launching the connection handlers in the
        background.

        :raises RuntimeError: If endpoint server has already been started.
        """
        self._logger.info("Starting endpoint server...")
        if self._started:
            raise RuntimeError("Endpoint server has already been started!")
        self._started = True

        self._connection_handler = threading.Thread(
            target=self._create_connection_handler, daemon=True
        )
        self._connection_handler.start()
        if self._c_timeout is not None:
            self._logger.info(
                "Connection timeout detected, starting periodic cleanup thread..."
            )
            self._connection_cleaner = threading.Thread(
                target=self._cleanup_connections, daemon=True
            )
            self._connection_cleaner.start()
        self._logger.info("Endpoint server started.")

    def stop(self, timeout=10):
        """Stops the endpoint server along all its connection endpoints, cleaning up
        underlying datastructures. This always shuts down all connection endpoints
        with a given timeout (see stop() of the Endpoint class for more information
        on this behavior).

        :param timeout: Allows each connection endpoint to process remaining messages
        until timeout.
        :raises RuntimeError: If endpoint server has not been started.
        """
        self._logger.info("Stopping endpoint server...")
        if not self._started:
            raise RuntimeError("Endpoint server has not been started!")
        self._started = False
        self._connection_handler.join()
        self._connection_cleaner.join()

        self._logger.info("Closing connections...")
        with self._c_lock:
            start = time()
            for endpoint in self._connections.values():
                threading.Thread(
                    target=lambda: endpoint.stop(shutdown=True, timeout=timeout),
                    daemon=True,
                ).start()
            sleep(max(0.0, timeout - time() - start))
            self._connections = {}
        self._logger.info("Endpoint server stopped.")

    def poll_connections(
        self,
    ) -> tuple[
        dict[tuple[str, int], StreamEndpoint], dict[tuple[str, int], StreamEndpoint]
    ]:
        """Polls the state of all current available connection endpoints, filtering
        them for readability and writability.

        Note that while this method is thread-safe in itself, it is not guaranteed
        that any returned endpoint will be still connected (and available) at the
        point of using it, since the underlying cleanup thread (if enabled) might
        have closed any potential dead endpoint if general timeout set (see __init__()).

        :return: Tuple of dictionary of addresses and endpoints from which can be
        read from / written to.
        """
        with self._c_lock:
            self._logger.debug(
                f"Polling {len(self._connections)} connections "
                "for readability and writability..."
            )
            c_states = {addr: (ep, ep.poll()) for addr, ep in self._connections.items()}
        r_ready = {addr: t[0] for addr, t in c_states.items() if t[1][0][1]}
        w_ready = {addr: t[0] for addr, t in c_states.items() if t[1][0][2]}
        self._logger.debug(
            f"{len(r_ready)} connections for readability and "
            f"{len(w_ready)} connections for writability found."
        )
        return r_ready, w_ready

    def get_connections(
        self, addrs: list[tuple[str, int]]
    ) -> dict[tuple[str, int], Optional[StreamEndpoint]]:
        """Checks a list of given client addresses whether there is an available
        connection endpoint for each of them and retrieves them.

        Note that while this method is thread-safe in itself, it is not guaranteed
        that any returned endpoint will be still connected (and available) at the
        point of using it, since the underlying cleanup thread (if enabled) might
        have closed any potential dead endpoint if general timeout set (see __init__()).

        :param addrs: Client addresses to check and retrieve endpoints for.
        :return: Dictionary of addresses and endpoints (None if not existing).
        """
        self._logger.debug(f"Trying to retrieve {len(addrs)} connections...")
        with self._c_lock:
            return {addr: self._connections.get(addr) for addr in addrs}

    def get_new_connections(
        self, n: int = 1, timeout: int = 10
    ) -> dict[tuple[str, int], StreamEndpoint]:
        """Checks and retrieves the first n new connections in the underlying queue
        filled by the connection handler.

        Note that while this method is thread-safe in itself, it is not guaranteed
        that any returned endpoint will be still connected (and available) at the
        point of using it, since the underlying cleanup thread (if enabled) might
        have closed any potential dead endpoint if general timeout set (see __init__()).

        :param n: Maximum numbers of new connections to retrieve.
        :param timeout: Maximum time when polling for new connections.
        :return: Dictionary of new addresses and endpoints of new connections.
        """
        self._logger.debug(f"Trying to retrieve {n} new connections...")
        new_connections = {}
        start = time()
        while self._started and len(new_connections) < n:
            try:
                remaining_timeout = max(0.0, timeout - time() - start)
                addr, ep = self._p_connections.get(timeout=remaining_timeout)
                new_connections[addr] = ep
            except queue.Empty:
                break
        self._logger.debug(
            f"{len(new_connections)} out of {n} new connections retrieved."
        )
        return new_connections

    def close_connections(self, addrs: list[tuple[str, int]], timeout: int = 10):
        """Checks a list of given client addresses whether there is an available
        connection endpoint for each of them and closes them, shutting them also down.

        :param addrs: Client addresses to check and close endpoints for.
        :param timeout: Collective timeout for shutting down connection endpoints.
        """
        self._logger.debug(f"Trying to close {len(addrs)} connections...")
        n = 0
        start = time()
        for addr in addrs:
            with self._c_lock:
                endpoint = self._connections.pop(addr, None)
            if endpoint is not None:
                self._logger.debug(f"Shutting down connection endpoint {addr}...")
                n += 1
                threading.Thread(
                    target=lambda: endpoint.stop(shutdown=True, timeout=timeout),
                    daemon=True,
                ).start()
        sleep(max(0.0, timeout - time() - start))
        self._logger.debug(f"{n} out of {len(addrs)} requested connections closed.")

    def _create_connection_handler(self):
        """Starts the loop to create new endpoints, connect them to their remote
        counterparts, and store them into the underlying datastructures of the server.
        """
        self._logger.info("AsyncHandler: Starting connection handler...")
        while self._started:
            self._n_connections += 1
            logging_prefix = f"AsyncHandler: [{self._n_connections}] "
            self._logger.debug(logging_prefix + "Preparing endpoint for connection...")
            new_connection = StreamEndpoint(
                name=f"{self._name}-{self._n_connections}",
                addr=self._addr,
                remote_addr=None,
                acceptor=True,
                send_b_size=self._send_b_size,
                recv_b_size=self._recv_b_size,
                compression=self._compression,
                marshal_f=self._marshal_f,
                unmarshal_f=self._unmarshal_f,
                multithreading=self._multithreading,
                buffer_size=self._buffer_size,
            )
            n_connection_rdy = new_connection.start(blocking=False)
            while self._started and not n_connection_rdy.wait(10):
                self._logger.debug(
                    logging_prefix + "Waiting for endpoint to establish a connection..."
                )
            if not self._started:
                break

            remote_addr = new_connection.poll()[1][1]
            while self._started:
                try:
                    self._logger.debug(
                        logging_prefix
                        + "Storing connection endpoint in pending queue..."
                    )
                    self._p_connections.put((remote_addr, new_connection), timeout=10)
                    with self._c_lock:
                        old_connection = self._connections.pop(remote_addr, None)
                    if old_connection is not None:
                        self._logger.debug(
                            logging_prefix
                            + "Shutting down existing connection endpoint..."
                        )
                        old_connection.stop(shutdown=True, timeout=10)
                    with self._c_lock:
                        self._connections[remote_addr] = new_connection
                    self._logger.debug(
                        logging_prefix + "New connection endpoint handled."
                    )
                    break
                except queue.Full:
                    self._logger.debug(
                        logging_prefix
                        + "Timeout triggered: Queue full. Discarding oldest endpoint..."
                    )
                    try:
                        self._p_connections.get(block=False)
                    except queue.Empty:
                        continue
        self._logger.info("AsyncHandler: Stopping...")

    def _cleanup_connections(self):
        """Starts the loop to periodically check all connection endpoints of the
        server for dead connections and clean up those that remain dead after a set
        timeout (see __init__()).
        """
        self._logger.info("AsyncCleaner: Starting periodic connection cleanup...")
        c_pending: dict[tuple[str, int], StreamEndpoint] = {}
        while self._started:
            with self._c_lock:
                c_dead = [addr for addr, ep in c_pending.items() if not ep.poll()[0][0]]
                self._logger.debug(
                    f"AsyncCleaner: Closing {len(c_dead)} out of {len(c_pending)} "
                    f"inactive and marked connection endpoints..."
                )
                self.close_connections(c_dead, timeout=0)
                c_pending = {
                    addr: ep
                    for addr, ep in self._connections.items()
                    if not ep.poll()[0][0]
                }
                self._logger.debug(
                    f"AsyncCleaner: {len(c_pending)} inactive connection endpoints "
                    "found and marked."
                )
            sleep(self._c_timeout)
        self._logger.info("AsyncCleaner: Stopping...")

    def __iter__(self):
        while self._started:
            try:
                yield self._p_connections.get(timeout=10)
            except queue.Empty:
                continue

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __del__(self):
        if self._started:
            self.stop()


def _convert_addr_to_name(addr: tuple) -> tuple[str, int]:
    """Translates a socket address, which is either a 2-tuple (ipv4) or a 4-tuple (ipv6)
    into a 2-tuple (host, port). Tries to resolve the host to its (DNS)
    hostname, otherwise keeps the numeric representation. Ports/Services are always
    kept numeric.

    :param addr: Address (ipv4/6) to convert.
    :return: Address tuple.
    """
    return socket.getnameinfo(addr, socket.NI_NUMERICSERV)[0], int(
        socket.getnameinfo(addr, socket.NI_NUMERICSERV)[1]
    )


# noinspection PyTypeChecker
def _send_payload(sock: socket.socket, payload: bytes):
    """Sends a payload over a socket, performing simple marshalling (size is sent
    first, then the bytes of the object). Blocking (if passed socket not configured
    otherwise).

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
    """Receives a payload over a socket, performing simple marshalling (size is
    received first, then the bytes of the object). Blocking (if passed socket not
    configured otherwise).

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
    :param buff_size: Maximum number of bytes to receive from socket per receive
    iteration.
    :return: Received n bytes.
    """
    data = bytearray(size)
    r_size = size
    while r_size > 0:
        n_data = sock.recv(min(r_size, buff_size))
        n_size = len(n_data)
        if n_size == 0:
            raise RuntimeError("Connection broken!")
        data[size - r_size : size - r_size + n_size] = n_data
        r_size -= n_size
    return data


def _check_r_socket(sock: socket.socket, timeout: int = None) -> bool:
    """Checks a given socket whether data can be read from it.

    :param sock: Socket to check for read readiness.
    :param timeout: Timeout (seconds) to wait for socket to be read ready.
    :return: True if socket is ready to be read from, else false.
    :raises TimeoutError: If timeout set and triggered.
    """
    if sock is None:
        return False
    if timeout is not None and not select.select([sock], [], [], timeout)[0]:
        raise TimeoutError
    elif not select.select([sock], [], [], 0)[0]:
        return False
    return True


def _check_w_socket(sock: socket.socket) -> bool:
    """Checks a given socket whether data can be written to it.

    :param sock: Socket to check for write readiness.
    :return: True if socket is ready to be written to, else false.
    """
    if sock is None:
        return False
    elif not select.select([], [sock], [], 0)[1]:
        return False
    return True


def _close_socket(sock: socket.socket):
    """Closes the socket of an endpoint, shutdowns any potential connection that
    might have been established.

    :param sock: Socket to close.
    """
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()
