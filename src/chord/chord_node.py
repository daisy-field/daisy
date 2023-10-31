"""
    Class for Federated Peer that implements a fault-tolerant version of the Chord Protocol

    Author: Lotta Fejzula
    Modified: 15.09.23



"""

import logging
from time import time
from uuid import uuid4
from enum import Enum
from typing import Self, Optional
import numpy as np
import typing

from src.communication.message_stream import StreamEndpoint, EndpointServer


class MessageType(Enum):
    JOIN = 1
    FIND_SUCC_RES = 2
    FIND_SUCC_REQ = 3
    STABILIZE = 4
    NOTIFY = 5


class MessageOrigin(Enum):
    PROCESS_JOIN = 1
    FIX_FINGERS = 2


class Chordmessage:
    """
    Class for Chord messages. Message_type can be chosen freely, but should be documented. Sender_id and Sender_addr
    will be set to the id and addr of the Node who sends the message. Payload can contain anything that should be
    processed by the receiving node in accordance to the message type.
    """
    message_id: uuid4
    message_type: MessageType
    payload: tuple[int, tuple[str, int]]

    def __init__(self, message_id: uuid4, message_type: MessageType,
                 payload: tuple[int, tuple[str, int]] = None):
        """
        Creates a new Chordmessage Object.

        :param message_type: denotes how the message will be processed at receiving endpoint
        :param sender_id: chord id of sending node
        :param sender_addr: chord address of sending node, used for replying
        :param payload: key-value pairs with the contents of the message
        """
        self.message_id = message_id
        self.message_type = message_type
        self.payload = payload


# TODO logging
# TODO Boostrapping
# TODO endpoint pro unique finger (max=maxfinger)
# TODO message id for tracking messages
# TODO use get_conn from eps here instead to check whether conn is already open
class Chordnode:
    """
    Class for Chordnodes.
    """

    _id: int
    _name: str  # for debugging and fun
    _addr: tuple[str, int]
    _finger_table: dict[int, tuple[int, tuple[str, int], StreamEndpoint]]
    _successor: tuple[int, tuple[str, int]]
    _successor_endpoint: StreamEndpoint
    _predecessor: tuple[int, tuple[str, int]]
    _predecessor_endpoint: StreamEndpoint
    _node_endpoints: list[StreamEndpoint]
    _endpoint_server: EndpointServer
    _max_fingers: int  # let m be the number of bits in the key/node identifiers. (copied from paper)
    _sent_messages: dict[uuid4, tuple[MessageOrigin, time, Optional[int]]]

    def __init__(self, name: str, addr: tuple[str, int],
                 successor: tuple[int, tuple[str, int]] = Self,
                 predecessor: tuple[int, tuple[str, int]] = None,
                 node_endpoints: list[StreamEndpoint] = None,
                 max_fingers: int = 32):
        """
        Creates a new Chord Peer.

        :param name: Name of peer for logging and fun.
        :param addr: Address of peer.
        :param finger_table: Dictionary of other nodes across the chord ring, known to this node.
        :param successor: Successor of node on Chord Ring, initialized as self.
        :param predecessor: Predecessor of node on Chord Ring, initialized as None.
        :param max_fingers: Max size of fingertable, also used for calculating peer ids on chord
        """

        self._id = hash(addr) % (2 ** max_fingers)
        self._name = name
        self._addr = addr
        self._finger_table = {}
        self._successor = successor  # init to None, set at join to chord ring
        self._predecessor = predecessor
        self._node_endpoints = node_endpoints
        self._endpoint_server = EndpointServer(f"{name}-endpointserver", addr=addr, multithreading=True)
        self._max_fingers = max_fingers

    def send_join(self, remote_addr: tuple[str, int]) -> None:
        """
        Sends join request to a node in an existing Chordring, or starts new.
        :param remote_addr: Address of bootstrap node
        """
        message_id = uuid4()
        self._sent_messages[message_id] = (MessageOrigin.PROCESS_JOIN, time(), None)

        join_chord_message = Chordmessage(message_id=message_id, message_type=MessageType.JOIN,
                                          payload=(self._id, self._addr))
        endpoint = StreamEndpoint(name=f"Join-Endpoint-{self._id}", addr=self._addr,
                                  remote_addr=remote_addr, acceptor=False, multithreading=True,
                                  buffer_size=10000)
        endpoint.start()
        endpoint.send(join_chord_message)
        endpoint.stop(shutdown=True)

    def send_notify(self, remote_id: int, remote_addr: tuple[str, int]) -> None:
        notify_message = Chordmessage(message_id=uuid4(), message_type=MessageType.NOTIFY,
                                      payload=self._predecessor)
        ep = self._check_for_endpoint(remote_id, remote_addr)
        self.send_message(ep, remote_addr, notify_message)

    def send_stabilize(self, remote_id: int, remote_addr: tuple[str, int]) -> None:
        if self._successor is None:
            raise RuntimeError(f"Stabilize on Successor Null on node {self._name}")
        stabilize_message = Chordmessage(message_id=uuid4(), message_type=MessageType.STABILIZE,
                                         payload=(self._id, self._addr))
        ep = self._check_for_endpoint(remote_id, remote_addr)
        self.send_message(ep, remote_addr, stabilize_message)

    def _fix_fingertable(self) -> None:
        """Function to keep fingertables current with nodes joining and leaving the chordring. Should be called
        periodically by each node.
        """
        for i in range(self._max_fingers):
            finger = self._id + 2 ** i % (2 ** self._max_fingers)
            if self._is_pred(finger):
                self._finger_table[i] = (self._successor[0], self._successor[1], self._successor_endpoint)
            else:
                message_id = uuid4()
                self._sent_messages[message_id] = (MessageOrigin.FIX_FINGERS, time(), i)
                self._find_succ(finger, self._addr, message_id)

    def _is_succ(self, node_id: int):
        return (self._id < self._predecessor[0]) & (node_id not in range(self._id + 1, self._predecessor[0] + 1)) \
            or (node_id in range(self._predecessor[0] + 1, self._id + 1))

    def _is_pred(self, node_id: int):
        return (self._id > self._successor[0]) & (node_id not in range(self._successor[0] + 1, self._id + 1)) \
            or (node_id in range(self._id + 1, self._successor[0] + 1))

    def _find_succ(self, node_id: int, remote_addr: tuple[str, int], message_id: uuid4) -> None:
        """Function to find successor of node with chord id find_succ_id. Message will be relayed along the cordring
        until the successor is found. In the end, the chord id and chord address of the found node will be sent back
        directly to the node who initially send the request.

        :param remote_addr: chord address of node who initially sent the find_successor request
        :param node_id: chord id of node whose successor should be found
        :message_id:
        """
        if self._is_pred(node_id):
            succ_found_msg = Chordmessage(message_id=message_id, message_type=MessageType.FIND_SUCC_RES,
                                          payload=self._successor)
            ep = self._check_for_endpoint(node_id, remote_addr)
            self.send_message(ep, remote_addr, succ_found_msg)

        elif self._is_succ(node_id):
            succ_found_msg = Chordmessage(message_id=message_id, message_type=MessageType.FIND_SUCC_RES,
                                          payload=(self._id, self._addr))
            ep = self._check_for_endpoint(node_id, remote_addr)
            self.send_message(ep, remote_addr, succ_found_msg)
        else:
            # idk, ask closest pred of node
            for finger in reversed(range(self._max_fingers)):
                if finger not in self._finger_table:
                    continue
                finger_node_id, finger_node_addr, finger_node_ep = self._finger_table[finger]
                if finger_node_id < node_id:  # TODO fix if with is_pred or is_succ logic
                    finger_node_ep.send(
                        Chordmessage(message_id=message_id, message_type=MessageType.FIND_SUCC_REQ,
                                     payload=(node_id, remote_addr)))
                    break

    def _check_for_endpoint(self, node_id, remote_addr) -> Optional[StreamEndpoint]:
        ep = {finger_ep for finger_id, finger_addr, finger_ep in self._finger_table.values()
              if node_id == finger_id}
        if len(ep) == 0:
            ep = self._endpoint_server.get_connections([remote_addr]).get(remote_addr)
        else:
            ep = ep.pop()

        return ep

    def send_message(self, ep, remote_addr, message):
        if ep is not None:
            ep.send(message)
        else:
            endpoint = StreamEndpoint(name=f"one-time-ep-{np.random.randint(0, 100)}",
                                      remote_addr=remote_addr, acceptor=False, multithreading=True,
                                      buffer_size=10000)
            endpoint.start()
            endpoint.send(message)
            endpoint.stop(shutdown=True)

    def _process_join(self, message: Chordmessage) -> None:
        self._find_succ(*message.payload, message.message_id)

    def _process_notify(self, message: Chordmessage) -> None:
        if message.payload[0] != self._id:
            self.send_stabilize(*message.payload)

    def _process_stabilize(self, message: Chordmessage) -> None:
        # if (node id is my pred or between me and my pred) update my pred
        # send notify with my pred
        if message.payload == self._predecessor or message.payload[0]:
            self.send_notify(*message.payload)

    def _process_find_succ_res(self, message: Chordmessage) -> None:

        if message.message_id not in self._sent_messages:
            return
        msg_origin, _, msg_finger = self._sent_messages[message.message_id]
        # succ from join
        if msg_origin == MessageOrigin.PROCESS_JOIN:
            self.send_stabilize(*message.payload)
        else:
            ep = self._check_for_endpoint(*message.payload)
            if ep is None:
                ep = StreamEndpoint(name=f"finger-ep-{np.random.randint(0, 100)}",
                                    remote_addr=message.payload[1], acceptor=False, multithreading=True,
                                    buffer_size=10000)
                ep.start()
            # TODO check if endpoint already exists in ft or is in useage or can be closed
            self._finger_table[msg_finger] = (message.payload[0], message.payload[1], ep)

    def _process_find_succ_req(self, message: Chordmessage) -> None:
        # call find succ with node from message as target
        self._find_succ(*message.payload, message.message_id)

    def run(self, bootstrap_addr: tuple[str, int] = None) -> None:
        self._endpoint_server.start()
        if bootstrap_addr:
            self.send_join(bootstrap_addr)

        while True:
            r_ready, _ = self._endpoint_server.poll_connections()
            for ep in r_ready:
                message = typing.cast(Chordmessage, r_ready[ep].receive(timeout=5))
                if message is not None:
                    msg_type = message.message_type
                    match msg_type:
                        case MessageType.JOIN:
                            self._process_join(message)
                        case MessageType.FIND_SUCC_RES:
                            self._process_find_succ_res(message)
                        case MessageType.FIND_SUCC_REQ:
                            self._process_find_succ_req(message)
                        case MessageType.STABILIZE:
                            self._process_stabilize(message)
                        case MessageType.NOTIFY:
                            self._process_notify(message)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)

    node = Chordnode(name="Serious Black", addr=("1.1.1.2", 21))
    node.run()
    node = Chordnode(name="Cordula Gruen", addr=("1.1.1.1", 21))
    node.run(("1.1.1.2", 21))
    pass
