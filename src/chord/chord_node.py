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


# TODO Comments :c

class MessageType(Enum):
    JOIN = 1
    FIND_SUCC_RES = 2
    FIND_SUCC_REQ = 3
    STABILIZE = 4
    NOTIFY = 5


class MessageOrigin(Enum):
    JOIN = 1
    FIX_FINGERS = 2


class Chordmessage:
    """
    Class for Chord messages. Message_type can be chosen freely, but should be documented. Sender_id and Sender_addr
    will be set to the id and addr of the Node who sends the message. Payload can contain anything that should be
    processed by the receiving node in accordance to the message type.
    """
    message_id: uuid4
    message_type: MessageType
    node_tuple: tuple[int, tuple[str, int]]

    def __init__(self, message_id: uuid4, message_type: MessageType,
                 node_tuple: tuple[int, tuple[str, int]] = None):
        """
        Creates a new Chordmessage Object.
        :param message_id: message identifier
        :param message_type: denotes how the message will be processed at receiving endpoint
        :param node_tuple:
        """
        self.message_id = message_id
        self.message_type = message_type
        self.node_tuple = node_tuple


# TODO Boostrapping
# TODO logging

class Chordnode:
    """
    Class for Chordnodes.
    """

    _id: int
    _name: str  # for debugging and fun
    _addr: tuple[str, int]
    _fingertable: dict[int, tuple[int, tuple[str, int], StreamEndpoint]]
    _successor: tuple[int, tuple[str, int]]
    _successor_endpoint: StreamEndpoint
    _predecessor: tuple[int, tuple[str, int]]
    _predecessor_endpoint: StreamEndpoint
    _node_endpoints: list[StreamEndpoint]
    _endpoint_server: EndpointServer
    _max_fingers: int  # let m/max_fingers be the number of bits in the key/node identifiers. (copied from paper)
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
        :param successor: Successor of node on Chord Ring, initialized as self.
        :param predecessor: Predecessor of node on Chord Ring, initialized as None.
        :param max_fingers: Max size of fingertable, also used for calculating peer ids on chord
        """

        self._id = hash(addr) % (2 ** max_fingers)
        self._name = name
        self._addr = addr
        self._fingertable = {}
        self._successor = successor  # init to None, set at join to chord ring
        self._predecessor = predecessor
        self._node_endpoints = node_endpoints
        self._endpoint_server = EndpointServer(f"{name}-endpointserver", addr=addr, multithreading=True)
        self._max_fingers = max_fingers

    def _send_message(self, ep, remote_addr, message):
        if ep is not None:
            if self._clean_up_ep_if_dropout(ep):
                return
            ep.send(message)
        else:
            endpoint = StreamEndpoint(name=f"one-time-ep-{np.random.randint(0, 100)}",
                                      remote_addr=remote_addr, acceptor=False, multithreading=True,
                                      buffer_size=10000)
            endpoint.start()
            endpoint.send(message)
            endpoint.stop(shutdown=True)

    def _join(self, remote_addr: tuple[str, int]):
        """
        Sends join request to node in existing Chordring
        :param remote_addr: Address of bootstrap node
        """
        if remote_addr is not None:
            message_id = uuid4()
            self._sent_messages[message_id] = (MessageOrigin.JOIN, time(), None)

            join_chord_message = Chordmessage(message_id=message_id, message_type=MessageType.JOIN,
                                              node_tuple=(self._id, self._addr))
            endpoint = StreamEndpoint(name=f"Join-Endpoint-{self._id}", addr=self._addr,
                                      remote_addr=remote_addr, acceptor=False, multithreading=True,
                                      buffer_size=10000)
            endpoint.start()
            endpoint.send(join_chord_message)
            endpoint.stop(shutdown=True)
        # if remote_addr is none start new chordring

    def _notify(self, remote_id: int, remote_addr: tuple[str, int]):
        notify_message = Chordmessage(message_id=uuid4(), message_type=MessageType.NOTIFY,
                                      node_tuple=self._predecessor)
        ep = self._get_ep_if_exists(remote_id, remote_addr)
        self._send_message(ep, remote_addr, notify_message)

    def _stabilize(self, remote_id: int, remote_addr: tuple[str, int]):
        stabilize_message = Chordmessage(message_id=uuid4(), message_type=MessageType.STABILIZE,
                                         node_tuple=(self._id, self._addr))
        ep = self._get_ep_if_exists(remote_id, remote_addr)
        self._send_message(ep, remote_addr, stabilize_message)

    def _update_finger(self, finger_node: tuple[int, tuple[str, int]], finger_id: int):
        finger = self._fingertable.pop(finger_id, None)
        if finger is not None:
            finger_ids_in_ft = {finger_id for finger_id, finger_addr, finger_ep in self._fingertable.values()
                                if finger[2] is finger_ep}
            if len(finger_ids_in_ft) == 0:
                self._close_and_remove_ep(finger[1], finger[2])

        ep = self._get_ep_if_exists(*finger_node)  # checks ep-server and self._finger_table
        if ep is not None:
            if self._clean_up_ep_if_dropout(ep):
                return
        else:  # create new ep if none exists
            ep = StreamEndpoint(name=f"finger-ep-{np.random.randint(0, 100)}",
                                remote_addr=finger_node[1], acceptor=False, multithreading=True,
                                buffer_size=10000)
            ep.start()
        self._fingertable[finger_id] = (finger_node[0], finger_node[1], ep)

    def _get_ep_if_exists(self, node_id: int, node_addr: tuple[str, int]) -> Optional[StreamEndpoint]:
        ep = {finger_ep for finger_id, finger_addr, finger_ep in self._fingertable.values()
              if node_id == finger_id}
        if len(ep) == 0:
            ep = self._endpoint_server.get_connections([node_addr]).get(node_addr)
        else:
            ep = ep.pop()
        return ep

    def _clean_up_ep_if_dropout(self, ep: StreamEndpoint) -> bool:
        """"
        true if ep is dropout
        """
        if ep is None:
            return True
        states, addrs = ep.poll()
        if states[0]:
            return False
        dropouts = [finger_key for finger_key, finger in self._fingertable.items()
                    if ep is finger[2]]
        for do in dropouts:
            do_id, do_addr, do_ep = self._fingertable.pop(do, (None, None, None))
            if do_ep is None:
                continue
            self._close_and_remove_ep(do_addr, do_ep)
        if ep is self._predecessor_endpoint:
            self._close_and_remove_ep(self._predecessor[1], ep)
            self._predecessor = None
            self._predecessor_endpoint = None
        if ep is self._successor_endpoint:
            self._close_and_remove_ep(self._successor[1], ep)
            for finger in range(self._max_fingers):
                f_id, f_addr, f_ep = self._fingertable.get(finger, (None, None, None))
                if not self._clean_up_ep_if_dropout(f_ep):
                    self._successor = (f_id, f_addr)
                    self._successor_endpoint = f_ep
                    break
            if self._successor is None:
                self._successor = self._predecessor
                self._successor_endpoint = self._predecessor_endpoint
        return True

    def _close_and_remove_ep(self, ep_addr: tuple[str, int], ep: StreamEndpoint):
        ep_in_epserver = self._endpoint_server.get_connections([ep_addr]).get(ep_addr)
        if ep_in_epserver is None and ep is not None:
            ep.stop(shutdown=True)
        else:
            self._endpoint_server.close_connections([ep_addr])

    def _fix_fingers(self):
        """Function to keep fingertables current with nodes joining and leaving the chordring. Should be called
        periodically by each node.
        """
        # TODO what if fingers die
        for i in range(self._max_fingers):
            finger = self._id + 2 ** i % (2 ** self._max_fingers)
            if self._is_pred(finger):
                self._update_finger(self._successor, i)
            else:
                message_id = uuid4()
                self._sent_messages[message_id] = (MessageOrigin.FIX_FINGERS, time(), i)
                self._find_succ(finger, self._addr, message_id)
                # fixme only returns succ of fingers, but what if finger doesnt change?

    def _is_succ(self, node_id: int) -> bool:
        return (self._id < self._predecessor[0]) & (node_id not in range(self._id + 1, self._predecessor[0] + 1)) \
            or (node_id in range(self._predecessor[0] + 1, self._id + 1))

    def _is_pred(self, node_id: int) -> bool:
        return (self._id > self._successor[0]) & (node_id not in range(self._successor[0] + 1, self._id + 1)) \
            or (node_id in range(self._id + 1, self._successor[0] + 1))

    def _get_closest_known_pred(self, node_id: int) -> StreamEndpoint | None:
        closest_pred = None
        for finger in range(self._max_fingers):
            f_curr = self._fingertable.get(finger, None)  # id, addr, ep
            f_next = self._fingertable.get(finger + 1, None)

            i = finger + 1
            while (f_curr is None or f_next is None) and i < self._max_fingers:
                if f_curr is None:
                    f_curr = self._fingertable.get(i, None)
                if f_next is None:
                    f_next = self._fingertable.get(i + 1, None)
                i += 1
                if f_curr[0] == f_next[0] and f_curr is not None:
                    f_next = self._fingertable.get(i, None)

            if (f_curr[0] < f_next[0]) & (node_id in range(f_curr[0], f_next[0] + 1)):
                return f_curr[2]  # FIXME
            if (f_curr[0] > f_next[0]) & (node_id in range(f_next[0], f_curr[0] + 1)):
                return f_next[2]
        return closest_pred

    def _find_succ(self, node_id: int, node_addr: tuple[str, int], message_id: uuid4):
        """Function to find successor of node with chord id find_succ_id. Message will be relayed along the cordring
        until the successor is found. In the end, the chord id and chord address of the found node will be sent back
        directly to the node who initially send the request.

        :param node_addr: chord address of node who initially sent the find_successor request
        :param node_id: chord id of node whose successor should be found
        :message_id:
        """
        if self._is_pred(node_id):
            succ_found_msg = Chordmessage(message_id=message_id, message_type=MessageType.FIND_SUCC_RES,
                                          node_tuple=self._successor)
            ep = self._get_ep_if_exists(node_id, node_addr)
            self._send_message(ep, node_addr, succ_found_msg)  # not closing ep if it comes from check

        elif self._is_succ(node_id):
            succ_found_msg = Chordmessage(message_id=message_id, message_type=MessageType.FIND_SUCC_RES,
                                          node_tuple=(self._id, self._addr))
            ep = self._get_ep_if_exists(node_id, node_addr)
            self._send_message(ep, node_addr, succ_found_msg)
        else:
            # idk, ask closest pred of node
            self._get_closest_known_pred(node_id).send(
                Chordmessage(message_id=message_id, message_type=MessageType.FIND_SUCC_REQ,
                             node_tuple=(node_id, node_addr)))

    def _process_join(self, message: Chordmessage):
        self._find_succ(*message.node_tuple, message.message_id)

    def _process_notify(self, message: Chordmessage):
        if message.node_tuple[0] != self._id:
            self._stabilize(*message.node_tuple)

    def _process_stabilize(self, message: Chordmessage):
        # if (node id is my pred (I am succ) or between me and my pred) update my pred
        if self._is_succ(message.node_tuple[0]):
            self._predecessor = message.node_tuple
            if self._get_ep_if_exists(*message.node_tuple) is None:
                self._predecessor_endpoint = StreamEndpoint(
                    name=f"{self._name}-predecessor-ep-{np.random.randint(0, 100)}",
                    remote_addr=self._addr, acceptor=False, multithreading=True,
                    buffer_size=10000)
            # TODO check current pred ep; close old ep and create new one if not exists
        # send notify with my pred
        if message.node_tuple == self._predecessor:
            self._notify(*message.node_tuple)

    def _process_find_succ_res(self, message: Chordmessage):
        if message.message_id not in self._sent_messages:
            return
        msg_origin, _, msg_finger = self._sent_messages.pop(message.message_id, (None, None, None))
        if msg_origin == MessageOrigin.JOIN:
            self._stabilize(*message.node_tuple)
        if msg_origin == MessageOrigin.FIX_FINGERS:
            self._update_finger(message.node_tuple, msg_finger)

    def _process_find_succ_req(self, message: Chordmessage):
        # call find succ with node from message as target
        self._find_succ(*message.node_tuple, message.message_id)

    def run(self, bootstrap_addr: tuple[str, int] = None):
        last_refresh_time = time()  # TODO implement dynamic ep creation relative to recv/time
        recv_messages_count = 0

        self._endpoint_server.start()
        self._join(bootstrap_addr)

        while True:
            curr_time = time()
            if curr_time - last_refresh_time >= 30:
                self._fix_fingers()
                self._stabilize(self._id, self._addr)
            r_ready, _ = self._endpoint_server.poll_connections()
            for ep in r_ready:
                message = typing.cast(Chordmessage, r_ready[ep].receive(timeout=5))
                if message is not None:
                    recv_messages_count += 1  # see task weiter oben
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
