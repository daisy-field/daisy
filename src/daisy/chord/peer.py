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
        multithreading=False,
        buffer_size=10000,
    )
    endpoint.start()
    endpoint.send(message)
    endpoint.stop(shutdown=True, timeout=15)


def receive_on_single_endpoint(endpoint: StreamEndpoint) -> list[Chordmessage]:
    """Receives and returns the Chordmessage of an endpoint.
    Defaults to an empty Chordmessage.

    :param endpoint: Endpoint to receive on.
    """
    chordmessages = []
    try:
        while True:
            chordmessages.append(typing.cast(Chordmessage, endpoint.receive(timeout=0)))
    except (RuntimeError, TimeoutError):
        return chordmessages


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
    # message id: ttl, send time, for fingers: i

    _logger: logging.Logger

    def __init__(
        self,
        peer_id: int,
        addr: tuple[str, int],
        successor: tuple[int, tuple[str, int]] = None,
        predecessor: tuple[int, tuple[str, int]] = None,
        max_fingers: int = 6,
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
            name="EndpointServer", addr=addr, multithreading=True, c_timeout=30
        )
        self._pending_requests = {}

        self._logger = logging.getLogger("Peer")

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
        self._pending_requests[request_id] = (60, time.time(), -1)
        self._logger.info(f"Sending join Request to {join_addr}")
        threading.Thread(target=lambda: send(join_addr, message), daemon=True).start()

    def _check_is_predecessor(self, check_id: int) -> bool:
        # check_id in ]pred, n[
        self._logger.info(f"Testing if {check_id} is a predecessor...")
        if self._predecessor is None:
            return True
        pred_id = self._predecessor[0]
        return (
            (self._id < pred_id) and (check_id not in range(self._id, pred_id + 1))
        ) or (check_id in range(pred_id + 1, self._id))

    def _check_is_successor(self, check_id: int) -> bool:
        # check_id in ]n, succ]
        self._logger.info(f"Testing if {check_id} is a successor...")
        succ_id = self._successor[0]
        if succ_id == self._id:
            # edge case: first join right after create ->
            # pred is still nil, and succ is inited to self
            return True
        return (
            (self._id > succ_id) and (check_id not in range(succ_id + 1, self._id + 1))
        ) or (check_id in range(self._id + 1, succ_id + 1))

    def _find_closest_predecessor(
        self, lookup_id: int
    ) -> tuple[int, tuple[str, int], StreamEndpoint] | None:
        if len(self._fingers) > 0:
            for i in reversed(range(0, self._max_fingers)):
                # ja, der check hier soll so implementiert sein,
                # to avoid predecessor null defaulting
                finger = self._fingers.get(i)
                if (
                    (lookup_id < self._id)
                    and finger[0] not in range(lookup_id, self._id + 1)
                ) or finger[0] in range(self._id + 1, lookup_id):
                    return finger
        return None

    def _lookup_peer_locally(self, lookup_id) -> tuple[int, tuple[str, int]] | None:
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
        a new lookup request will be made to the succeeding peer.

        :param lookup_id: ID of a peer whose successor shouöd be loooked up
        :param response_addr: Address of the peer who should receive the lookup response
        """
        result = self._lookup_peer_locally(lookup_id)
        if result is None:
            self._logger.info(f"Relaying lookup to {self._successor[0]}...")
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
            self._logger.info(f"Replying to lookup with result {result}...")
            message = Chordmessage(
                message_type=MessageType.LOOKUP_RES,
                sender=(self._id, self._addr),
                peer_tuple=result,
                request_id=request_id,
                origin=lookup_origin,
            )
            threading.Thread(
                target=lambda: send(response_addr, message), daemon=True
            ).start()

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
        threading.Thread(
            target=lambda: send(notify_peer[1], notify_message), daemon=True
        ).start()

    def _fix_fingers(self):
        self._logger.info("Initiating Fingertable updates, prepare for logging spam...")
        for i in range(self._max_fingers):
            finger = (self._id + (1 << i)) % (1 << self._max_fingers)
            request_id = uuid4()
            self._pending_requests[request_id] = (10, time.time(), i)
            self._lookup(finger, self._addr, MessageOrigin.FIX_FINGERS, request_id)

    def _check_finger_is_unique(
        self, finger: tuple[int, tuple[str, int], StreamEndpoint]
    ):
        finger_count = 0
        for finger_id, finger_addr, finger_ep in self._fingers.values():
            if finger[0] == finger_id:
                finger_count += 1
        if finger_count > 1:
            self._logger.critical(f"finger {finger[0]} is not unique")
            return False
        self._logger.critical(f"finger {finger[0]} is unique")
        return True

    def _get_endpoint_if_peer_already_in_fingers(
        self, index: int, new_finger: tuple[int, tuple[str, int]]
    ) -> StreamEndpoint | None:
        """checks whether a peer is already in the fingertable of another peer and
        returns its corresponding endpoint.

        :param index: index (also called key) of the peer on the fingertable
        :param new_finger: new finger tuple, containing a peers id and address
        :return: endpoint or None
        """
        for key in self._fingers:
            finger = self._fingers.get(key)
            if finger[0] == new_finger[0] and index != key:
                self._logger.critical(
                    f"reusing endpoint of finger {key}, for finger {index}"
                )
                return finger[2]
        self._logger.critical("not reusing endpoint")
        return None

    def _set_finger(self, index: int, new_finger: tuple[int, tuple[str, int]]):
        finger = self._fingers.get(index)
        if finger is not None:
            if (
                finger[0] != new_finger[0]
            ):  # only change entry if finger has actally changed
                if self._check_finger_is_unique(finger=finger):
                    # only stop existing endpoint if its not used somewhere else
                    finger[2].stop(shutdown=True)

                finger_endpoint = self._get_endpoint_if_peer_already_in_fingers(
                    index=index, new_finger=new_finger
                )
                if finger_endpoint is None:
                    # ep does not already exist
                    finger_endpoint = StreamEndpoint(
                        name=f"FINGER{new_finger[0]}",
                        remote_addr=new_finger[1],
                        acceptor=False,
                        multithreading=False,
                        buffer_size=10000,
                    )
                    finger_endpoint.start()

                self._fingers[index] = (
                    new_finger[0],
                    new_finger[1],
                    finger_endpoint,
                )
                self._logger.critical(
                    f"Updated Finger at Index {index} with {new_finger}..."
                )

        else:
            # finger has not at all existed before and is created completely new
            new_endpoint = StreamEndpoint(
                name=f"FINGER{new_finger[0]}",
                remote_addr=new_finger[1],
                acceptor=False,
                multithreading=False,
                buffer_size=10000,
            )
            new_endpoint.start()
            self._fingers[index] = (
                new_finger[0],
                new_finger[1],
                new_endpoint,
            )
            self._logger.critical(
                f"Created Finger at Index {index} with {new_finger}..."
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
        # TODO wo werden finger genutzt? -> lookup
        # TODO wie wird ttl von nachrichten gesetzt und geprüft?
        #  Was ist eine sinnvolle ttl?
        # TODO nachrichten mit ttl für alle nodes, sodass lookups die zu lange dauern
        #  abgebrochen werden können und nicht bis zum ende durchlaufen nur
        #  um dann nicht genutt zu werden
        # TODO retry von verlorenen anfragen
        # TODO be faster
        self._logger.info(f"Peer {self._id} started...")
        start = time.time()
        period = start
        if join_addr is None:
            self._create()
        else:
            self._join(join_addr=join_addr)
        while True:
            self._close_ring_at_first_peer()
            self._cleanup_pending_requests()
            period = self._maintain_chord_ring(start, period)
            r_ready = self._get_read_ready_endpoints()
            if len(r_ready) == 0:
                sleep(1)
                continue
            for ep in r_ready:
                for message in receive_on_single_endpoint(ep):
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

    def _close_ring_at_first_peer(self):
        if self._successor == (self._id, self._addr) and self._predecessor is not None:
            self._set_successor(self._predecessor)

    def _maintain_chord_ring(self, start: float, current_period: float) -> float:
        """wrapper for periodic stabilize and fix finger calls. Handles the timimng of maintenance calls.

        :param start: startup time of the peer
        :param current_period: bginn of the last stabilization period
        :return: starting time of the current or the new stabilization period
        """
        now = time.time()
        # set stabilization interval according to startup time of node
        stabilize_interval = 30
        if start - now < 30:  # 300 -> 5 minutes up and running
            stabilize_interval = 5
        # check whether new period should be started
        if now - current_period > stabilize_interval:
            # conduct maintenance calls
            if self._successor is not None and self._successor != (
                self._id,
                self._addr,
            ):
                self._stabilize()
                self._fix_fingers()
            return now
        return current_period

    def _cleanup_pending_requests(self):
        """Checks whether pending requests have spoiled and removes them."""
        delete_count = 0
        for key in list(self._pending_requests):
            request = self._pending_requests.get(key, None)
            if request is not None and time.time() - request[1] > request[0]:
                self._pending_requests.pop(key)
                delete_count += 1
        self._logger.critical(
            f"Deleted {delete_count} pending requests in this interation."
        )

    def _process_notify(self, message: Chordmessage):
        if self._check_is_successor(message.peer_tuple[0]):
            self._set_successor(message.peer_tuple)

    def _process_stabilize(self, message: Chordmessage):
        if self._check_is_predecessor(message.peer_tuple[0]):
            self._set_predecessor(message.peer_tuple)
        self._notify(message.peer_tuple)

    def _process_lookup_request(self, message: Chordmessage):
        self._logger.info(f"Received lookup request from {message.origin}...")
        self._lookup(
            *message.peer_tuple,
            lookup_origin=message.origin,
            request_id=message.id,
        )

    def _process_join_request(self, message: Chordmessage):
        self._logger.info(f"Received join from {message.peer_tuple[0]}...")
        self._lookup(
            *message.peer_tuple,
            lookup_origin=message.origin,
            request_id=message.id,
        )

    def _process_lookup_response(self, message: Chordmessage):
        self._logger.info(f"Received lookup response from {message.origin}...")
        # check whether request has spoiled
        request = self._pending_requests.get(message.id, None)
        if request is None:
            return None
        self._pending_requests.pop(message.id)
        # handle valid requests
        if message.origin == MessageOrigin.FIX_FINGERS:
            self._logger.info("Received finger from FIx-Fingers request...")
            self._set_finger(request[2], message.peer_tuple)
        if message.origin == MessageOrigin.JOIN:
            self._logger.info("Received successor from Join...")
            self._set_successor(message.peer_tuple)

    def _set_successor(self, successor: tuple[int, tuple[str, int]]):
        """Setter for a peer's successor. Assigns new successor and
        establishes new endpoint. Shuts down the old Endpoint.

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
        """Setter for a peer's predecessor. Assigns new predecessor and
        establishes new endpoint.Shuts down the old Endpoint.

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
            + f"\t\t\t\tpending requests: {len(self._pending_requests)}\n".expandtabs(
                10
            )
        )

    def _fingertable_to_string(self):
        return [
            (
                finger,
                ((self._id + (1 << finger)) % (1 << self._max_fingers)),
                self._fingers[finger][0],
                self._fingers[finger][2].name,
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
