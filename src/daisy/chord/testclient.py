# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import threading
import time
import typing
from enum import Enum
from time import sleep
from uuid import uuid4

from daisy.communication import EndpointServer, StreamEndpoint


def close_tmp_ep(ep: StreamEndpoint, sleep_time: int = 10, stop_timeout: int = 10):
    sleep(sleep_time)
    ep.stop(shutdown=True, timeout=stop_timeout)


class MessageType(Enum):
    JOIN = 1
    LOOKUP_RES = 2
    LOOKUP_REQ = 3
    STABILIZE = 4
    NOTIFY = 5


class MessageOrigin(Enum):
    FED_PEERS_REQ = 3
    JOIN = 1
    FIX_FINGERS = 2


class Chordmessage:
    """Class for Chord messages."""

    id: uuid4
    type: MessageType
    peer_tuple: tuple[int, tuple[str, int]]
    origin: MessageOrigin
    sender: tuple[int, tuple[str, int]]
    timestamp: float

    def __init__(
        self,
        request_id: uuid4 = None,
        message_type: MessageType = None,
        peer_tuple: tuple[int, tuple[str, int]] = None,
        origin: MessageOrigin = None,
        sender: tuple[int, tuple[str, int]] = None,
        timestamp: float = 0,
    ):
        """Creates a new Chordmessage.
        :param request_id: Message identifier
        :param message_type: Type of message for processing in receive function.
        :param peer_tuple: ID and address of the peer sent whithin the Chordmessage.
        :param origin: Origin of the Chordmessage.
        :param sender: Sender of the Chordmessage.
        """
        self.id = request_id
        self.type = message_type
        self.peer_tuple = peer_tuple
        self.origin = origin
        self.sender = sender
        self.timestamp = timestamp


class TestPeer:
    """Class for Chordpeers."""

    _id: int
    _name: str  # for debugging and fun
    _addr: tuple[str, int]
    _endpoint_server: EndpointServer

    def __init__(self, name: str, addr: tuple[str, int], id_test: int = None):
        """
        Creates a new Chord Peer.

        :param name: Name of peer for logging and fun.
        :param addr: Address of peer.
        """

        self._id = id_test  # hash(addr) % (2 ** max_fingers)
        self._name = name
        self._addr = addr
        self._endpoint_server = self._endpoint_server = EndpointServer(
            name="EndpointServer",
            addr=addr,
            multithreading=True,
            c_timeout=60,
            keep_alive=False,
        )

    def send_lookup(self, remote_addr: tuple[str, int]):
        id = int(input("enter lookup value"))
        message = Chordmessage(
            message_type=MessageType.LOOKUP_REQ,
            peer_tuple=(id, self._addr),
            request_id=123,
            origin=MessageOrigin.FED_PEERS_REQ,
            timestamp=time.time(),
        )
        ep_name = f"ep-{id}"
        endpoint = StreamEndpoint(
            name=ep_name,
            remote_addr=remote_addr,
            acceptor=False,
            multithreading=False,
            buffer_size=10000,
        )
        endpoint.start()
        endpoint.send(message)

        threading.Thread(
            target=lambda: close_tmp_ep(endpoint, 0, 20), daemon=True
        ).start()
        sleep(5)
        self.receive()

    def receive(self):
        while True:
            r_ready, _ = self._endpoint_server.poll_connections()
            for adr in r_ready:
                ep = r_ready[adr]
                recv = typing.cast(Chordmessage, ep.receive(timeout=0))
                if recv is not None:
                    print(recv.peer_tuple)
                    print(recv.sender)
                    return
                sleep(2)

    def start(self):
        self._endpoint_server.start()


if __name__ == "__main__":
    testpeer = TestPeer("localhost", ("127.0.0.1", 15888), id_test=1)
    testpeer.start()
    addr = input("enter target port: ")
    testpeer.send_lookup(("127.0.0.1", int(addr)))
    while True:
        cmd = input("enter command:")
        match cmd:
            case "send":
                addr = input("enter target port: ")
                testpeer.send_lookup(("127.0.0.1", int(addr)))
            case "exit":
                break
