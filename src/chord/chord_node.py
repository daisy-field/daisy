"""
    Class for Federated Peer that implements a fault-tolerant version of the Chord Protocol

    Author: Lotta Fejzula
    Modified: 15.09.23
"""

import logging

from typing import Self
from src.communication.message_stream import StreamEndpoint


# TODO implement recv message handler
# TODO logging
# TODO Boostrapping

# util stuffs
def node_in_range_through_zero(start: int, end: int, boundary: int, node_id: int):
    """
    util function that checks whether node_id is between start and end on the chord ring

    :start: startpoint of range, may be bigger than end
    :end: endpoint of range, is included in
    :boundary: splitting point of ranges if 0 is passed in chord ring
    :node_id: id for lookup
    """
    if start > end:  # range passes 0, interval right open
        return (node_id in range(start, boundary + 1)) or (node_id in range(0, end+1))
    else:  # not
        return node_id in range(start, end+1)


class Chordnode:
    """
    Class for Chordnodes.
    """

    _id: int
    _name: str
    _addr: tuple[str, int]
    _finger_table: {int: StreamEndpoint}
    _successor: tuple[int, StreamEndpoint]
    _predecessor: tuple[int, StreamEndpoint]  # int is id of succ/pred
    _node_recv_endpoint: StreamEndpoint

    def __init__(self, node_id: int = 0, name: str = "unnamed", addr: tuple[str, int] = ("127.0.0.1", 8080),
                 finger_table: {int: StreamEndpoint} = None,
                 successor: tuple[int, StreamEndpoint] = Self, predecessor: tuple[int, StreamEndpoint] = None,
                 node_recv_endpoint: StreamEndpoint = None):
        """
        Creates a new Chord Peer.

        :param node_id: Name of endpoint for logging purposes.
        :param name: Name of Peer.
        :param addr: Address of Peer.
        :param finger_table: Dictionary of other nodes across the chord ring, known to this node.
        :param successor: Successor of node on Chord Ring, initialized as self
        :param predecessor: Predecessor of node on Chord Ring, initialized as None
        """

        self._id = node_id
        self._name = name
        self._addr = addr
        self._finger_table = finger_table  # starts at 1
        self._successor = successor  # init to None, set at join to chord ring
        self._predecessor = predecessor
        self._node_recv_endpoint = node_recv_endpoint

    def open_chord(self):
        # open accepting endpoint for joining nodes
        self._node_recv_endpoint = StreamEndpoint(name=f"Acceptor-Public-{Chordnode._name}", addr=self._addr,
                                                  acceptor=True, multithreading=False, buffer_size=10000)
        self._node_recv_endpoint.start()
        while True:
            try:
                print(f"{self._name}-{self._node_recv_endpoint.start().receive(5)}")
            except TimeoutError:
                print(f"{self._name}-Public Endpoint opened, pls join!")

    def join_chord(self, remote_addr: tuple[str, int]):
        """
        Function to join existing Chordring.
        :param remote_addr: Address of node to send the join message to
        """
        # TODO implement
        # open listening endpoint to receive successor
        self._node_recv_endpoint = StreamEndpoint(name=f"Acceptor-Public-{Chordnode._name}", addr=self._addr,
                                                  acceptor=True, multithreading=False, buffer_size=10000)
        self._node_recv_endpoint.start()
        # send find_succ
        self.find_successor(self._addr, self._id)
        # receive
        # notify

    def notify(self, remote_node_addr: tuple[str, int], remote_node_id: int):
        # TODO implement
        if (self._predecessor is None) or node_in_range_through_zero(self._predecessor[0], self._id, 1, 1):
            pass

    def stabilize(self):
        # TODO implement
        pass

    def fix_fingertable(self):
        """
        Function to keep fingertables current with nodes joining and leaving the chordring. Should be called
        periodically by each node.
        """
        for i in range(self._finger_table.len()):
            self.find_successor(self._addr, self._id + 2 ** i)

    def find_successor(self, sender_addr: tuple[str, int], find_succ_id: int):
        """
        Function to find successor(find_succ_id). If current node does no know successor(find_succ_id) it
        will ask its successor. If successor(find_succ_id) is found, its id and addr will be sent back directly to the
        node who initially send the request.
        :param sender_addr:
        :param find_succ_id:
        """
        # find succ passes 0 on ring, makes this ugly af
        if (self._id < self._successor[0]) & (find_succ_id not in range(self._successor[0], self._id + 1)) \
                or (find_succ_id in range(self._id, self._successor[0] + 1)):
            # add successor info to chordmessage
            succ_found_msg = Chordmessage(message_type="find_successor_success", sender_id=self._id,
                                          sender_addr=self._addr, payload={"successor": self._successor})
            # check whether find_succ_id is in fingertable or open new endpoint
            if self._finger_table.get(find_succ_id):
                self._finger_table.get(find_succ_id).send(succ_found_msg)
            else:
                endpoint = StreamEndpoint(name=f"Sender-find_successor-id:{find_succ_id}", addr=self._addr,
                                          remote_addr=sender_addr, acceptor=False, multithreading=False,
                                          buffer_size=10000)
                endpoint.start()
                endpoint.send(succ_found_msg)
                endpoint.stop(shutdown=True)  # ?
        else:  # if IDK -> ask node that closest precedes find_succ_id
            for finger in self._finger_table.keys():
                if self._id < self._finger_table.get(finger) < find_succ_id:
                    self._finger_table.get(finger).send(
                        Chordmessage(message_type="find_successor", sender_id=self._id, sender_addr=self._addr,
                                     payload={"find_id": find_succ_id, "reply_addr": sender_addr}))
                    break


class Chordmessage:
    """
    Class for Chord messages. Message_type can be chosen freely, but should be documented. Sender_id and Sender_addr
    will be set to the id and addr of the Node who sends the message. Payload can contain anything that should be
    processed by the receiving node in accordance to the message type.
    """
    _message_type: str
    _sender_id: int
    _sender_addr: tuple[str, int]
    _payload: dict

    def __init__(self, message_type: str = "unspecified", sender_id: int = -1, sender_addr: tuple[str, int] = None,
                 payload: dict = None):
        self._message_type = message_type
        self._sender_id = sender_id
        self._sender_addr = sender_addr
        self._payload = payload


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    pass
