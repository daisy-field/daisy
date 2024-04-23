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
    peer_tuple: tuple[int, tuple[str, int]]
    origin: MessageOrigin
    sender: tuple[int, tuple[str, int]]

    def __init__(
        self,
        request_id: uuid4 = None,
        message_type: MessageType = None,
        peer_tuple: tuple[int, tuple[str, int]] = None,
        origin: MessageOrigin = None,
        sender: tuple[int, tuple[str, int]] = None,
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


def send(send_addr: tuple[str, int], message: Chordmessage):
    endpoint: StreamEndpoint = StreamEndpoint(
        name=f"SEND{message.sender[0]}",
        remote_addr=send_addr,
        acceptor=False,
        multithreading=True,
        buffer_size=10000,
    )
    endpoint.start()
    endpoint.send(message)
    threading.Thread(target=lambda: endpoint.stop(shutdown=True), daemon=True).start()


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
            "EndpointServer", addr=addr, multithreading=True, c_timeout=30
        )
        self._pending_requests = {}

        self._logger = logging.getLogger("Peer")

    def _create(self):
        self._logger.critical("Creating new Chord Ring...")
        self._endpoint_server.start()
        self._successor = (
            self._id,
            self._addr,
        )
        self._logger.critical(f"Listening on address {self._addr}")

    def _join(self, join_addr: tuple[str, int]):
        self._logger.critical("Joining existig Chord Ring...")
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
        self._logger.critical(f"Sending join Request to {join_addr}")
        send(join_addr, message)

    def _check_is_predecessor(self, check_id: int) -> bool:
        # check_id in ]pred, n[
        self._logger.critical(f"Testing if {check_id} is a predecessor...")
        if self._predecessor is None:
            return True
        pred_id = self._predecessor[0]
        return (
            (self._id < pred_id) and (check_id not in range(self._id, pred_id + 1))
        ) or (check_id in range(pred_id + 1, self._id))

    def _check_is_successor(self, check_id: int) -> bool:
        # check_id in ]n, succ]
        self._logger.critical(f"Testing if {check_id} is a successor...")
        succ_id = self._successor[0]
        if succ_id == self._id:
            # edge case: first join right after create ->
            # pred is still nil, and succ is inited to self
            return True
        return (
            (self._id > succ_id) and (check_id not in range(succ_id + 1, self._id + 1))
        ) or (check_id in range(self._id + 1, succ_id + 1))

    def _lookup_peer_locally(self, lookup_id) -> tuple[int, tuple[str, int]] | None:
        """Searches through a peers local data(-structures) to find whether
        the successor of a given peer is known locally.

        :param lookup_id: ID of a peer on the chord ring
        """
        self._logger.critical(f"Conducting local lookup for {lookup_id}...")
        if self._predecessor is not None and lookup_id == self._predecessor[0]:
            return self._predecessor
        elif self._successor is not None and self._check_is_successor(lookup_id):
            return self._successor
        elif self._check_is_predecessor(lookup_id):
            return self._id, self._addr
        else:
            return None

    def _find_closest_predecessor(
        self, lookup_id: int
    ) -> tuple[int, tuple[str, int], StreamEndpoint] | None:
        if len(self._fingers) > 0:
            for i in range(0, self._max_fingers, -1):
                # ja, der check hier soll so implementiert sein,
                # damit das predecessor null defaulting weg ist
                if (
                    (self._id > lookup_id)
                    and self._fingers[i][0] in range(self._id + 1, lookup_id)
                ) or self._fingers[i][0] not in range(lookup_id, self._id + 1):
                    return self._fingers[i]
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
        a new lookup request will be made to the succeeding peer.

        :param lookup_id: ID of a peer whose successor shouöd be loooked up
        :param response_addr: Address of the peer who should receive the lookup response
        """
        result = self._lookup_peer_locally(lookup_id)
        if result is None:
            forward_message = Chordmessage(
                message_type=MessageType.LOOKUP_REQ,
                sender=(self._id, self._addr),
                peer_tuple=(lookup_id, response_addr),
                request_id=request_id,
                origin=lookup_origin,
            )
            finger = self._find_closest_predecessor(lookup_id)
            if finger is None:
                self._logger.critical("Forwarding lookup to successor...")
                self._successor_endpoint.send(forward_message)
                return
            self._logger.critical(f"Forwarding lookup to finger {finger}...")
            finger[2].send(forward_message)
            return
        else:
            self._logger.critical(f"Replying to lookup with result {result}...")
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
        self._logger.critical(f"Stabilizing {self._successor}...")
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
        self._logger.critical(f"Notifying {notify_peer}...")
        send(notify_peer[1], notify_message)

    def _fix_fingers(self):
        self._logger.critical(
            "Initiating Fingertable updates, prepare for logging spam..."
        )
        for i in range(self._max_fingers):
            finger = self._id + (1 << i) % (1 << self._max_fingers)
            request_id = uuid4()
            self._pending_requests[request_id] = (10, time.time(), i)
            self._lookup(finger, self._addr, MessageOrigin.FIX_FINGERS, request_id)

    def _set_finger(self, index: int, new_finger_tuple: tuple[int, tuple[str, int]]):
        finger = self._fingers.get(index)
        self._logger.critical(
            f"Setting Finger at Index {index} to {new_finger_tuple}..."
        )
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
        self._logger.critical("Collecting all readable Endpoints...")
        r_ready_eps = set()
        if self._successor_endpoint and self._successor_endpoint.poll()[0][1]:
            r_ready_eps.add(self._successor_endpoint)
            self._logger.critical("Successor endpoint is readable...")
        if self._predecessor_endpoint and self._predecessor_endpoint.poll()[0][1]:
            r_ready_eps.add(self._predecessor_endpoint)
            self._logger.critical("Predecessor endpoint is readable...")
        self._logger.critical("Polling EndpointServer...")
        r_ready, _ = self._endpoint_server.poll_connections()
        for addr in r_ready:
            r_ready_eps.add(r_ready[addr])
        self._logger.critical(f"Found {len(r_ready_eps)} readable endpoints...")
        return r_ready_eps

    def run(self, join_addr: tuple[str, int] = None):
        # TODO wo werden finger genutzt?
        # TODO wie wird ttl von nachrichten gesetzt und geprüft?
        # TODO retry von verlorenen anfragen
        self._logger.critical(f"Peer {self._id} started...")

        if join_addr is None:
            self._create()
        else:
            self._join(join_addr=join_addr)
        while True:
            self._maintain_chord_ring()
            sleep(1)
            r_ready = self._get_read_ready_endpoints()
            for ep in r_ready:
                message = self._receive_on_single_endpoint(ep)
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
            self._logger.critical(self.__str__())

    def _maintain_chord_ring(self):
        """wrapper for periodic stabilize and fix finger calls. Also closes the ring
        in case only two peers are present.
        """
        if self._successor == (self._id, self._addr) and self._predecessor is not None:
            self._set_successor(self._predecessor)
        if self._successor is not None and self._successor != (
            self._id,
            self._addr,
        ):
            self._stabilize()
            self._fix_fingers()

    def _receive_on_single_endpoint(
        self, endpoint: StreamEndpoint
    ) -> Chordmessage | None:
        """Helper Method; Receives and return the Chrdmessage of one endpoint. Defaults to an empty CHordmessage.

        :param endpoint: Endpoint to receive on.
        """
        try:
            return typing.cast(Chordmessage, endpoint.receive(timeout=0))
        except RuntimeError:
            self._logger.warning("Endpoint has not been started!")
        return Chordmessage()

    def _process_notify(self, message: Chordmessage):
        if self._check_is_successor(message.peer_tuple[0]):
            self._set_successor(message.peer_tuple)

    def _process_stabilize(self, message: Chordmessage):
        if self._check_is_predecessor(message.peer_tuple[0]):
            self._set_predecessor(message.peer_tuple)
        self._notify(message.peer_tuple)

    def _process_lookup_request(self, message: Chordmessage):
        self._logger.critical(f"Received lookup request from {message.origin}...")
        self._logger.critical("Initiating successor lookup...")
        self._lookup(
            *message.peer_tuple,
            lookup_origin=message.origin,
            request_id=message.id,
        )

    def _process_join_request(self, message: Chordmessage):
        self._logger.critical(f"Received join from {message.peer_tuple[0]}...")
        self._logger.critical("Initiating successor lookup...")
        self._lookup(
            *message.peer_tuple,
            lookup_origin=message.origin,
            request_id=message.id,
        )

    def _process_lookup_response(self, message: Chordmessage):
        self._logger.critical(f"Received lookup response from {message.origin}...")
        if message.origin == MessageOrigin.JOIN:
            self._logger.critical("Received successor from Join...")
            self._pending_requests.pop(message.id)
            self._set_successor(message.peer_tuple)
        if message.origin == MessageOrigin.FIX_FINGERS:
            self._logger.critical("Received finger from FIx-Fingers request...")
            i = self._pending_requests.pop(message.id)[2]
            self._set_finger(i, message.peer_tuple)

    def _set_successor(self, successor: tuple[int, tuple[str, int]]):
        """Setter method for a peer's successor. Assigns new successor and
        establishes new endpoint.

        :param successor: id and address of new successor
        """
        self._logger.critical(f"Setting new Successor {successor}...")
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
        self._logger.critical(f"Setting new Predecessor {predecessor}...")
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
        level=logging.CRITICAL,  # handlers=[
        #    logging.FileHandler(
        #        filename=f"./logging/peer-{args.id}.log",
        #        mode="w",
        #        encoding="utf8",
        #    ),
        #    logging.StreamHandler(),
        # ],
    )

    localhost = "127.0.0.1"

    peer = Peer(peer_id=args.id, addr=(localhost, args.port))
    if args.joinPort:
        peer.run((localhost, args.joinPort))  # start as first chord peer
    else:
        peer.run()
