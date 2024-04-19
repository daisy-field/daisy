# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import argparse
import threading
import typing
from enum import Enum
from time import sleep
from uuid import uuid4

from daisy.communication import EndpointServer, StreamEndpoint


def shutdown_temporary_endpoint(
    ep: StreamEndpoint, sleep_time: int = 10, stop_timeout: int = 10
):
    sleep(sleep_time)
    ep.stop(shutdown=True, timeout=stop_timeout)


class MessageType(Enum):
    JOIN = 1
    LOOKUP_RES = 2
    LOOKUP_REQ = 3
    STABILIZE = 4
    NOTIFY = 5


class MessageOrigin(Enum):
    JOIN = 1
    FIX_FINGERS = 2


class Chordmessage:
    """Class for Chord messages."""

    id: uuid4
    type: MessageType
    sender: tuple[int, tuple[str, int]]
    peer_tuple: tuple[int, tuple[str, int]]
    origin: MessageOrigin

    def __init__(
        self,
        message_id: uuid4,
        message_type: MessageType = None,
        sender: tuple[int, tuple[str, int]] = None,
        peer_tuple: tuple[int, tuple[str, int]] = None,
        origin: MessageOrigin = None,
    ):
        """Creates a new Chordmessage.
        :param message_id: Message identifier
        :param message_type: Type of message for processing in receive function.
        :param peer_tuple: ID and address of the peer sent whithin the Chordmessage.
        """
        self.id = message_id
        self.type = message_type
        self.sender = sender
        self.peer_tuple = peer_tuple
        self.origin = origin


def send(send_addr: tuple[str, int], message: Chordmessage):
    endpoint: StreamEndpoint = StreamEndpoint(
        name="tmp",
        remote_addr=send_addr,
        acceptor=False,
        multithreading=True,
        buffer_size=10000,
    )
    endpoint.start()
    endpoint.send(message)
    threading.Thread(
        target=lambda: shutdown_temporary_endpoint(endpoint, 10, 10), daemon=True
    ).start()


class Peer:
    _id: int
    _addr: tuple[str, int]

    _successor: tuple[int, tuple[str, int]] | None
    _successor_endpoint: StreamEndpoint | None
    _predecessor: tuple[int, tuple[str, int]] | None
    _predecessor_endpoint: StreamEndpoint | None

    _endpoint_server: EndpointServer

    _max_fingers: int

    def __init__(
        self,
        p_id: int,
        addr: tuple[str, int],
        successor: tuple[int, tuple[str, int]] = None,
        predecessor: tuple[int, tuple[str, int]] = None,
        max_fingers: int = 16,
    ):
        self._id = p_id
        self._addr = addr
        self._successor = successor
        self._successor_endpoint = None
        self._predecessor = predecessor
        self._predecessor_endpoint = None
        self._endpoint_server = EndpointServer(
            f"{id}-Server", addr=addr, multithreading=True, c_timeout=30
        )
        self._max_fingers = max_fingers

    def _create(self):
        self._endpoint_server.start()
        self._successor = (
            self._id,
            self._addr,
        )

    def _join(self, join_addr: tuple[str, int]):
        self._endpoint_server.start()
        message = Chordmessage(
            message_type=MessageType.JOIN,
            sender=(self._id, self._addr),
            peer_tuple=(self._id, self._addr),
            message_id=1,
        )
        send(join_addr, message)

    def _check_is_predecessor(self, check_id: int) -> bool:
        # check_id in ]pred, n[
        if self._predecessor is None:
            return True
        pred_id = self._predecessor[0]
        return (
            (self._id < pred_id) and (check_id not in range(self._id, pred_id + 1))
        ) or (check_id in range(pred_id + 1, self._id))

    def _check_is_successor(self, check_id: int) -> bool:
        # check_id in [n, succ]
        succ_id = self._successor[0]
        if succ_id == self._id:
            # edge case: first join right after create ->
            # pred is still nil, and succ is inited to self
            return True

        val = (
            (self._id > succ_id) and (check_id not in range(succ_id + 1, self._id + 1))
        ) or (check_id in range(self._id + 1, succ_id + 1))

        return val

    def _try_to_find_lookup_result_locally(
        self, lookup_id
    ) -> tuple[int, tuple[str, int]] | None:
        """Searches through a peers local data(-structures) to find whether
        the successor of a given peer is known locally.

        :param lookup_id: ID of a peer on the chord ring
        """
        if self._predecessor is not None and lookup_id == self._predecessor[0]:
            return self._predecessor
        elif self._successor is not None and self._check_is_successor(lookup_id):
            return self._successor
        elif self._check_is_predecessor(lookup_id):
            return self._id, self._addr
        else:
            return None

    def _lookup(
        self,
        lookup_id: int,
        response_addr: tuple[str, int],
        lookup_origin: MessageOrigin,
    ):
        """Initiaes local lookup of a peer's successor and attempts to
        send the found result back. If no maching peer is found locally,
        a new lookup request will  be made to the succeeding peer.

        :param lookup_id: ID of a peer whose successor shouöd be loooked up
        :param response_addr: Address of the peer who should receive the lookup response
        """
        result = self._try_to_find_lookup_result_locally(lookup_id)
        if result is None:
            self._successor_endpoint.send(
                Chordmessage(
                    message_type=MessageType.LOOKUP_REQ,
                    sender=(self._id, self._addr),
                    peer_tuple=(lookup_id, response_addr),
                    message_id=1,
                    origin=lookup_origin,
                )
            )
            return
        message = Chordmessage(
            message_type=MessageType.LOOKUP_RES,
            sender=(self._id, self._addr),
            peer_tuple=result,
            message_id=1,
            origin=lookup_origin,
        )
        send(response_addr, message)

    def _stabilize(self):
        """Creates and sends a stabilize message to a peer's successor"""
        stabilize_message = Chordmessage(
            message_type=MessageType.STABILIZE,
            sender=(self._id, self._addr),
            peer_tuple=(self._id, self._addr),
            message_id=1,
        )
        self._successor_endpoint.send(stabilize_message)

    def _notify(self, notify_peer: tuple[int, tuple[str, int]]):
        """Wrappermethod that creates and initiates a notify message

        :param notify_peer: recipient of the notify message
        """
        notify_message = Chordmessage(
            message_type=MessageType.NOTIFY,
            sender=(self._id, self._addr),
            peer_tuple=self._predecessor,
            message_id=1,
        )
        send(notify_peer[1], notify_message)

    def _fix_fingers(self):
        for i in range(self._max_fingers):
            finger = self._id + (1 << i) % (1 << self._max_fingers)
            self._lookup(
                finger, self._addr, MessageOrigin.FIX_FINGERS
            )  # dict(sorted(x.items(), key=lambda item: item[1])) TODO sort fingers

    def _get_read_ready_endpoints(self) -> set[StreamEndpoint]:
        r_ready_eps = set()
        if self._successor_endpoint and self._successor_endpoint.poll()[0][1]:
            r_ready_eps.add(self._successor_endpoint)
        if self._predecessor_endpoint and self._predecessor_endpoint.poll()[0][1]:
            r_ready_eps.add(self._predecessor_endpoint)
        # get r_ready eps from ep server
        r_ready, _ = self._endpoint_server.poll_connections()
        for addr in r_ready:
            r_ready_eps.add(r_ready[addr])
        return r_ready_eps

    def run(self, join_addr: tuple[str, int] = None):
        if join_addr is None:
            self._create()
        else:
            self._join(join_addr=join_addr)

        while True:
            r_ready = self._get_read_ready_endpoints()
            if (
                self._successor == (self._id, self._addr)
                and self._predecessor is not None
            ):
                self._set_successor(self._predecessor)
            if self._successor is not None and self._successor != (
                self._id,
                self._addr,
            ):
                self._stabilize()

            sleep(1)
            for ep in r_ready:
                message = typing.cast(Chordmessage, ep.receive(timeout=0))
                match message.type:
                    case MessageType.LOOKUP_REQ:
                        print(f"Lookup request for: {message.peer_tuple}")
                        self._lookup(*message.peer_tuple, lookup_origin=message.origin)
                    case MessageType.LOOKUP_RES:
                        if message.origin == MessageOrigin.JOIN:
                            self._set_successor(message.peer_tuple)
                        if message.origin == MessageOrigin.FIX_FINGERS:
                            pass  # TODO implement
                    case MessageType.JOIN:
                        self._lookup(*message.peer_tuple, lookup_origin=message.origin)
                    case MessageType.STABILIZE:
                        if self._check_is_predecessor(message.peer_tuple[0]):
                            self._set_predecessor(message.peer_tuple)
                        self._notify(message.peer_tuple)
                    case MessageType.NOTIFY:
                        if self._check_is_successor(message.peer_tuple[0]):
                            self._set_successor(message.peer_tuple)
            print(self.__str__())

    def _set_successor(self, successor: tuple[int, tuple[str, int]]):
        """Setter method for a peer's successor. Assigns new successor and
        establishes new endpoint.

        :param successor: id and address of new successor
        """
        self._successor = successor
        if self._successor_endpoint is not None:
            self._successor_endpoint.stop(shutdown=True)
        self._successor_endpoint = StreamEndpoint(
            name=f"succ-ep-{self._id}",
            remote_addr=successor[1],
            acceptor=False,
            multithreading=False,
            buffer_size=10000,
        )
        self._successor_endpoint.start()

    def _set_predecessor(self, predecessor: tuple[int, tuple[str, int]]):
        """Setter method for a peer's predecessor. Assigns new predecessor and
        establishes new endpoint.

        :param predecessor: id and address of new predecessor
        """
        self._predecessor = predecessor
        if self._predecessor_endpoint is not None:
            self._predecessor_endpoint.stop(shutdown=True)
        self._predecessor_endpoint = StreamEndpoint(
            name=f"pred-ep-{self._id}",
            remote_addr=predecessor[1],
            acceptor=False,
            multithreading=False,
            buffer_size=10000,
        )
        self._predecessor_endpoint.start()

    def __str__(self):
        return (
            f"id: {self._id}, \n"
            + f"predecessor: {self._predecessor}, \n"
            + f"successor: {self._successor}, \n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--id", type=int, default=None, help="Id of peer")
    parser.add_argument("--port", type=int, default=None, help="Port of peer")
    parser.add_argument("--succId", type=int, default=None, help="successor id")
    parser.add_argument("--succPort", type=int, default=None, help="successor port")
    parser.add_argument("--predId", type=int, default=None, help="predecessor tuple")
    parser.add_argument("--predPort", type=int, default=None, help="predecessor port")
    parser.add_argument("--joinPort", type=int, default=None, help="join port")

    args = parser.parse_args()

    localhost = "127.0.0.1"

    peer = Peer(p_id=args.id, addr=(localhost, args.port))
    if args.joinPort:
        peer.run((localhost, args.joinPort))  # start as first chord peer
    else:
        peer.run()
