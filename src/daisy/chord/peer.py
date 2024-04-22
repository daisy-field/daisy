# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import argparse
import logging
import threading
import time
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
        request_id: uuid4 = None,
        message_type: MessageType = None,
        sender: tuple[int, tuple[str, int]] = None,
        peer_tuple: tuple[int, tuple[str, int]] = None,
        origin: MessageOrigin = None,
    ):
        """Creates a new Chordmessage.
        :param request_id: Message identifier
        :param message_type: Type of message for processing in receive function.
        :param peer_tuple: ID and address of the peer sent whithin the Chordmessage.
        """
        self.id = request_id
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

    _max_fingers: int
    _fingers: dict[int, tuple[int, tuple[str, int], StreamEndpoint]]
    # i: id, addr, ep

    _endpoint_server: EndpointServer
    _pending_requests: dict[uuid4, tuple[int, float, int]]

    _logger: logging.Logger

    # message id: ttl, send time, for fingers: i

    def __init__(
        self,
        peer_id: int,
        addr: tuple[str, int],
        successor: tuple[int, tuple[str, int]] = None,
        predecessor: tuple[int, tuple[str, int]] = None,
        max_fingers: int = 16,
    ):
        self._id = peer_id
        self._addr = addr

        self._successor = successor
        self._successor_endpoint = None
        self._predecessor = predecessor
        self._predecessor_endpoint = None

        self._max_fingers = max_fingers
        self._fingers = {}

        self._endpoint_server = EndpointServer(
            f"{self._id}-Server", addr=addr, multithreading=True, c_timeout=30
        )
        self._pending_requests = {}

        self._logger = logging.getLogger(f"{self._id}-LOGGER")
        self._logger.setLevel(logging.INFO)

    def _create(self):
        self._logger.info("Creating new Chord Ring...")
        self._endpoint_server.start()
        self._successor = (
            self._id,
            self._addr,
        )
        self._logger.info(f"Listening on address {self._addr}")

    def _join(self, join_addr: tuple[str, int]):
        self._logger.info("Joining existig Chord Ring...")
        self._endpoint_server.start()
        request_id = uuid4()
        message = Chordmessage(
            message_type=MessageType.JOIN,
            sender=(self._id, self._addr),
            peer_tuple=(self._id, self._addr),
            origin=MessageOrigin.JOIN,
            request_id=request_id,
        )
        self._pending_requests[request_id] = (10, time.time(), -1)
        self._logger.info(f"Sending join Request to {join_addr}")
        send(join_addr, message)

    def _check_is_predecessor(self, check_id: int) -> bool:
        # check_id in ]pred, n[
        self._logger.info(f"Testing wether {check_id} is a predecessor...")
        if self._predecessor is None:
            return True
        pred_id = self._predecessor[0]
        return (
            (self._id < pred_id) and (check_id not in range(self._id, pred_id + 1))
        ) or (check_id in range(pred_id + 1, self._id))

    def _check_is_successor(self, check_id: int) -> bool:
        # check_id in ]n, succ]
        self._logger.info(f"Testing wether {check_id} is a successor...")
        succ_id = self._successor[0]
        if succ_id == self._id:
            # edge case: first join right after create ->
            # pred is still nil, and succ is inited to self
            return True
        return (
            (self._id > succ_id) and (check_id not in range(succ_id + 1, self._id + 1))
        ) or (check_id in range(self._id + 1, succ_id + 1))

    def _try_to_find_lookup_result_locally(
        self, lookup_id
    ) -> tuple[int, tuple[str, int]] | None:
        """Searches through a peers local data(-structures) to find whether
        the successor of a given peer is known locally.

        :param lookup_id: ID of a peer on the chord ring
        """
        self._logger.info(f"Conducting local lookup for {lookup_id}...")
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
        request_id: uuid4,
    ):
        """Initiaes local lookup of a peer's successor and attempts to
        send the found result back. If no maching peer is found locally,
        a new lookup request will  be made to the succeeding peer.

        :param lookup_id: ID of a peer whose successor shouöd be loooked up
        :param response_addr: Address of the peer who should receive the lookup response
        """
        result = self._try_to_find_lookup_result_locally(lookup_id)
        if result is None:
            self._logger.info("Forwarding lookup to successor...")
            self._successor_endpoint.send(
                Chordmessage(
                    message_type=MessageType.LOOKUP_REQ,
                    sender=(self._id, self._addr),
                    peer_tuple=(lookup_id, response_addr),
                    request_id=request_id,
                    origin=lookup_origin,
                )
            )
            return
        self._logger.info(f"Replying to lookup with result {result}...")
        message = Chordmessage(
            message_type=MessageType.LOOKUP_RES,
            sender=(self._id, self._addr),
            peer_tuple=result,
            request_id=request_id,
            origin=lookup_origin,
        )
        send(response_addr, message)

    def _stabilize(self):
        """Creates and sends a stabilize message to a peer's successor"""
        stabilize_message = Chordmessage(
            message_type=MessageType.STABILIZE,
            sender=(self._id, self._addr),
            peer_tuple=(self._id, self._addr),
        )
        self._logger.info(f"Stabilizing {self._successor}...")
        self._successor_endpoint.send(stabilize_message)

    def _notify(self, notify_peer: tuple[int, tuple[str, int]]):
        """Creates and initiates a notify message

        :param notify_peer: recipient of the notify message
        """
        notify_message = Chordmessage(
            message_type=MessageType.NOTIFY,
            sender=(self._id, self._addr),
            peer_tuple=self._predecessor,
        )
        self._logger.info(f"Notifying {notify_peer}...")
        send(notify_peer[1], notify_message)

    def _fix_fingers(self):
        self._logger.info("Initiating Fingertable updates, prepare for logging spam...")
        for i in range(self._max_fingers):
            finger = self._id + (1 << i) % (1 << self._max_fingers)
            request_id = uuid4()
            self._pending_requests[request_id] = (10, time.time(), i)
            self._lookup(finger, self._addr, MessageOrigin.FIX_FINGERS, request_id)

    def _set_finger(self, index: int, new_finger_tuple: tuple[int, tuple[str, int]]):
        finger = self._fingers.get(index)
        self._logger.info(f"Setting Finger at Index {index} to {new_finger_tuple}...")
        if finger is not None:
            if finger[0] != new_finger_tuple[0]:
                finger[2].stop(shutdown=True)
            self._fingers[index] = (
                new_finger_tuple[0],
                new_finger_tuple[1],
                StreamEndpoint(
                    name=f"FINGER-{index}",
                    remote_addr=new_finger_tuple[1],
                    acceptor=False,
                    multithreading=False,
                    buffer_size=10000,
                ),
            )
        else:
            self._fingers[index] = (
                new_finger_tuple[0],
                new_finger_tuple[1],
                StreamEndpoint(
                    name=f"FINGER-{index}",
                    remote_addr=new_finger_tuple[1],
                    acceptor=False,
                    multithreading=False,
                    buffer_size=10000,
                ),
            )

    def _get_read_ready_endpoints(self) -> set[StreamEndpoint]:
        self._logger.info("Collecting all readable Endpoints...")
        r_ready_eps = set()
        if self._successor_endpoint and self._successor_endpoint.poll()[0][1]:
            r_ready_eps.add(self._successor_endpoint)
            self._logger.info("Successor endpoint is readable...")
        if self._predecessor_endpoint and self._predecessor_endpoint.poll()[0][1]:
            r_ready_eps.add(self._predecessor_endpoint)
            self._logger.info("Predecessor endpoint is readable...")
        self._logger.info("Polling EndpointServer...")
        r_ready, _ = self._endpoint_server.poll_connections()
        for addr in r_ready:
            r_ready_eps.add(r_ready[addr])
        self._logger.info(f"Found {len(r_ready_eps)} readable endpoints...")
        return r_ready_eps

    def run(self, join_addr: tuple[str, int] = None):
        # TODO wo werden finger genutzt?
        # TODO wie wird ttl von nachrichten gesetzt und geprüft?
        # TODO retry von verlorenen anfragen
        self._logger.info(f"Peer {self._id} started...")

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
                self._fix_fingers()
            sleep(1)
            for ep in r_ready:
                message = typing.cast(Chordmessage, ep.receive(timeout=0))
                match message.type:
                    case MessageType.LOOKUP_REQ:
                        self._process_lookup_request(message)
                    case MessageType.LOOKUP_RES:
                        self._process_lookup_response(message)
                    case MessageType.JOIN:
                        self._process_join_request(message)
                    case MessageType.STABILIZE:
                        self._process_stabilize(message)
                    case MessageType.NOTIFY:
                        self._process_notify(message)
            self._logger.info(self.__str__())

    def _process_notify(self, message):
        if self._check_is_successor(message.peer_tuple[0]):
            self._set_successor(message.peer_tuple)

    def _process_stabilize(self, message):
        if self._check_is_predecessor(message.peer_tuple[0]):
            self._set_predecessor(message.peer_tuple)
        self._notify(message.peer_tuple)

    def _process_lookup_request(self, message):
        self._logger.info(f"Received lookup request from {message.origin}...")
        self._logger.info("Initiating successor lookup...")
        self._lookup(
            *message.peer_tuple,
            lookup_origin=message.origin,
            request_id=message.id,
        )

    def _process_join_request(self, message):
        self._logger.info(f"Received join from {message.peer_tuple[0]}...")
        self._logger.info("Initiating successor lookup...")
        self._lookup(
            *message.peer_tuple,
            lookup_origin=message.origin,
            request_id=message.id,
        )

    def _process_lookup_response(self, message):
        self._logger.info(f"Received lookup response from {message.origin}...")
        if message.origin == MessageOrigin.JOIN:
            self._logger.info("Received successor from Join...")
            self._pending_requests.pop(message.id)
            self._set_successor(message.peer_tuple)
        if message.origin == MessageOrigin.FIX_FINGERS:
            self._logger.info("Received finger from FIx-Fingers request...")
            i = self._pending_requests.pop(message.id)[2]
            self._set_finger(i, message.peer_tuple)

    def _set_successor(self, successor: tuple[int, tuple[str, int]]):
        """Setter method for a peer's successor. Assigns new successor and
        establishes new endpoint.

        :param successor: id and address of new successor
        """
        self._logger.info(f"Setting new Successor {successor}...")
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
        self._logger.info(f"Setting new Predecessor {predecessor}...")
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

    def __str__(self) -> str:
        return (
            "Peer status: \n"
            + f"\t\t\t\tid: {self._id}, \n".expandtabs(10)
            + f"\t\t\t\tpredecessor: {self._predecessor}, \n".expandtabs(10)
            + f"\t\t\t\tsuccessor: {self._successor}, \n".expandtabs(10)
            + f"\t\t\t\t{self._fingertable_to_string()}\n".expandtabs(10)
        )

    def _fingertable_to_string(self):
        return [
            (
                finger,
                self._id + (1 << finger) % (1 << self._max_fingers),
                self._fingers[finger][0],
            )
            for finger in self._fingers
        ]


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

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                filename=f"./logging/peer-{args.id}.log",
                mode="w",
                encoding="utf8",
            ),
            logging.StreamHandler(),
        ],
    )

    localhost = "127.0.0.1"

    peer = Peer(peer_id=args.id, addr=(localhost, args.port))
    if args.joinPort:
        peer.run((localhost, args.joinPort))  # start as first chord peer
    else:
        peer.run()
