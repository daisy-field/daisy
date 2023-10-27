"""
    Class for Federated Peer that implements a fault-tolerant version of the Chord Protocol

    Author: Lotta Fejzula
    Modified: 15.09.23
"""

import logging
from enum import Enum
from typing import Self
import numpy as np
import typing

from src.communication.message_stream import StreamEndpoint, EndpointServer


class MessageType(Enum):
    JOIN = 1
    FIND_SUCC_RES = 2
    FIND_SUCC_REQ = 3
    STABILIZE = 4
    NOTIFY = 5


class Chordmessage:
    """
    Class for Chord messages. Message_type can be chosen freely, but should be documented. Sender_id and Sender_addr
    will be set to the id and addr of the Node who sends the message. Payload can contain anything that should be
    processed by the receiving node in accordance to the message type.
    """
    message_type: MessageType
    _sender_id: int
    _sender_addr: tuple[str, int]
    payload: dict


    def __init__(self, message_type: MessageType, sender_id: int = -1, sender_addr: tuple[str, int] = None,
                 payload: dict = None):
        """
        Creates a new Chordmessage Object.

        :param message_type: denotes how the message will be processed at receiving endpoint
        :param sender_id: chord id of sending node
        :param sender_addr: chord address of sending node, used for replying
        :param payload: key-value pairs with the contents of the message
        """
        self.message_type = message_type
        self._sender_id = sender_id
        self._sender_addr = sender_addr
        self.payload = payload


# TODO implement recv message handler
# TODO logging
# TODO Boostrapping

class Chordnode:
    """
    Class for Chordnodes.
    """

    _id: int
    _name: str  # for debugging
    _addr: tuple[str, int]
    _finger_table: dict[int: [tuple[str, int], StreamEndpoint]]
    _successor: tuple[int, tuple[str, int]]
    _successor_endpoint: StreamEndpoint
    _predecessor: tuple[int, tuple[str, int]]
    _predecessor_endpoint: StreamEndpoint
    _node_endpoints: list[StreamEndpoint]
    _endpoint_server: EndpointServer
    _max_fingers: int

    def __init__(self, name: str, addr: tuple[str, int],
                 finger_table: dict[int: [tuple[str, int], StreamEndpoint]] = None,
                 successor: tuple[int, tuple[str, int]] = Self,
                 predecessor: tuple[int, tuple[str, int]] = None,
                 node_endpoints: list[StreamEndpoint] = None,
                 max_fingers: int = 32):
        """
        Creates a new Chord Peer.

        :param name: Name of Peer.
        :param addr: Address of Peer.
        :param finger_table: Dictionary of other nodes across the chord ring, known to this node.
        :param successor: Successor of node on Chord Ring, initialized as self
        :param predecessor: Predecessor of node on Chord Ring, initialized as None
        """

        self._id = hash(addr) % (2 ^ max_fingers)
        self._name = name
        self._addr = addr
        self._finger_table = finger_table  # starts at 1
        self._successor = successor  # init to None, set at join to chord ring
        self._predecessor = predecessor
        self._node_endpoints = node_endpoints
        self._endpoint_server = EndpointServer("", addr=addr, multithreading=True)
        self._max_fingers = max_fingers

    def send_join(self, remote_addr: tuple[str, int]) -> None:
        """
        Sends join request to a node in an existing Chordring, or starts new.
        :param remote_addr: Address of bootstrap node
        """

        join_chord_message = Chordmessage(message_type=MessageType.JOIN, sender_id=self._id,
                                          sender_addr=self._addr, payload={"node": (self._id, self._addr)})
        endpoint = StreamEndpoint(name=f"Join-Endpoint-{self._id}", addr=self._addr,
                                  remote_addr=remote_addr, acceptor=False, multithreading=False,
                                  buffer_size=10000)
        endpoint.start()
        endpoint.send(join_chord_message)
        endpoint.stop(shutdown=True)

    def send_notify(self, remote_id: int, remote_addr: tuple[str, int]) -> None:
        notify_message = Chordmessage(message_type=MessageType.NOTIFY, sender_id=self._id, sender_addr=self._addr,
                                      payload={"predecessor": self._predecessor})
        node_in_ft = self._finger_table.get(remote_id)
        if node_in_ft is not None:
            node_in_ft.send(notify_message)
        else:
            endpoint = StreamEndpoint(name=f"Join-Endpoint-{self._id}", addr=self._addr,
                                      remote_addr=remote_addr, acceptor=False, multithreading=False,
                                      buffer_size=10000)
            endpoint.start()
            endpoint.send(notify_message)

    def send_stabilize(self, remote_id: int, remote_addr: tuple[str, int]) -> None:
        if self._successor is None:
            raise RuntimeError(f"Stabilize on Successor Null on node {self._name}")

        stabilize_message = Chordmessage(message_type=MessageType.STABILIZE, sender_id=self._id, sender_addr=self._addr,
                                         payload={"node": (self._id, self._addr)})

        node_in_ft = self._finger_table.get(remote_id)
        if node_in_ft is not None:
            node_in_ft.send(stabilize_message)
        else:
            endpoint = StreamEndpoint(name=f"Stabilize-Endpoint-{self._id}", addr=self._addr,
                                      remote_addr=remote_addr, acceptor=False, multithreading=False,
                                      buffer_size=10000)
            endpoint.start()
            endpoint.send(stabilize_message)

    def _fix_fingertable(self) -> None:
        """Function to keep fingertables current with nodes joining and leaving the chordring. Should be called
        periodically by each node.
        """
        for i in range(self._max_fingers):
            # todo wont work like this
            self._find_succ(self._id + 2 ** i % self._max_fingers, self._addr)

    def _find_succ(self, node_id: int, remote_addr: tuple[str, int]) -> None:
        # todo update predecessor if necessary
        # this should work, stop thinking about it!
        """Function to find successor of node with chord id find_succ_id. Message will be relayed along the cordring
        until the successor is found. In the end, the chord id and chord address of the found node will be sent back
        directly to the node who initially send the request.
        :param remote_addr: chord address of node who initially sent the find_successor request
        :param node_id: chord id of node whose successor should be found
        """
        if (self._id > self._successor[0]) & (node_id not in range(self._successor[0]+1, self._id + 1)) \
                or (node_id in range(self._id+1, self._successor[0] + 1)):
            # add successor info to chordmessage
            succ_found_msg = Chordmessage(message_type=MessageType.FIND_SUCC_RES, sender_id=self._id,
                                          sender_addr=self._addr, payload={"succ": self._successor})
            # todo check whether ep to find_succ_id exists instead here and
            node_in_ft = self._finger_table.get(node_id)
            if node_in_ft is not None:
                node_in_ft.send(succ_found_msg)
            else:
                endpoint = StreamEndpoint(name=f"{self._name}-{self._id}-send-succ", addr=self._addr,
                                          remote_addr=remote_addr, acceptor=False, multithreading=False,
                                          buffer_size=10000)
                endpoint.start()
                endpoint.send(succ_found_msg)
                endpoint.stop(shutdown=True)
            return

        if node_id in range(self._predecessor[0], self._id+1):  # return self if new node is predecessor
            succ_found_msg = Chordmessage(message_type=MessageType.FIND_SUCC_RES, sender_id=self._id,
                                          sender_addr=self._addr, payload={"succ": (self._id, self._addr)})
            # here
            node_in_ft = self._finger_table.get(node_id)
            if node_in_ft is not None:
                node_in_ft.send(succ_found_msg)

            else:
                endpoint = StreamEndpoint(name=f"{self._name}-{self._id}-send-succ", addr=self._addr,
                                          remote_addr=remote_addr, acceptor=False, multithreading=False,
                                          buffer_size=10000)
                endpoint.start()
                endpoint.send(succ_found_msg)
                endpoint.stop(shutdown=True)
            return
        # if IDK -> ask node that closest precedes node_id
        for finger in self._finger_table.keys():
            if self._id < self._finger_table.get(finger) < node_id:
                self._finger_table.get(finger).send(
                    Chordmessage(message_type=MessageType.FIND_SUCC_REQ, sender_id=self._id, sender_addr=self._addr,
                                 payload={"find_id": node_id, "reply_addr": remote_addr}))
                # TODO message id for tracking
                break

    def _process_join(self, message: Chordmessage) -> None:
        self._find_succ(message.payload.get("node")[0], message.payload.get("node")[1])

    def _process_notify(self, message: Chordmessage) -> None:
        if message.payload.get("predecessor")[0] != self._id:
            self.send_stabilize(message.payload.get("predecessor")[0], message.payload.get("predecessor")[1])

    def _process_stabilize(self, message: Chordmessage) -> None:
        # todo implement
        # if (node id is my pred or between me and my pred) update my pred
        # send notify with my pred
        if message.payload.get("node") == self._predecessor:  # or node betwen me and pred
            self.send_notify(message.payload.get("node")[0], message.payload.get("node")[1])

    def _process_find_succ_res(self, message: Chordmessage) -> None:
        # todo implement
        # stabilize(succ from message)
        pass

    def _process_find_succ_req(self, message: Chordmessage) -> None:
        # todo implement
        # call find succ with node from message as target
        self._find_succ(message.payload.get("id"), message.payload.get("addr"))

    def _open_endpoints(self, count: int = 1) -> None:
        for i in range(count):
            self._node_endpoints.append(
                StreamEndpoint(name=f"{self._name}-{self._id}-recv-{i}-{np.random.randint(0, 100)}", addr=self._addr,
                               acceptor=True, multithreading=False, buffer_size=10000))

    def _select_endpoints(self) -> None:
        # todo implement
        pass

    def _close_endpoints(self) -> None:
        # todo implement
        pass

    def _run(self, bootstrap_addr: tuple[str, int]) -> None:
        # todo implement multiple endpoints
        self._endpoint_server.start()
        self.send_join(bootstrap_addr)

        while True:
            message = typing.cast(Chordmessage, self._node_endpoints.pop(0).receive(5))
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
    node = Chordnode(1, "cordula gruen", ("1.1.1.1", 21))
    node.run(("1.1.1.2", 21))
    pass
