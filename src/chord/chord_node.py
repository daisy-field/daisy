"""
    Class for Federated Peer that implements a fault-tolerant version of the Chord Protocol

    Author: Lotta Fejzula
    Modified: 15.09.23
"""
import logging
from typing import Optional, Self
# Bootstrap-problem :c

from src.communication.message_stream import StreamEndpoint


# TODO implement receive message handler
# TODO implement receiving find succ request in receive message handler

class Chordnode:
    """
    Class for creating nodes for the Chord DHT
    """

    _id: int

    _name: str
    _addr: tuple[str, int]
    _remote_addr: tuple[str, int]

    _finger_table: {int: StreamEndpoint}  # dict?
    # int is id of node
    # Save endpoints for lookup in ft instead of address tuples
    _successor: tuple[int, StreamEndpoint]
    _predecessor: tuple[int, StreamEndpoint]  # int is id of succ/pred

    def __init__(self, node_id: int = 0, name: str = "unnamed", addr: tuple[str, int] = ("127.0.0.1", 8080),
                 remote_addr: tuple[str, int] = None, finger_table: {int: StreamEndpoint} = None,
                 successor: tuple[int, StreamEndpoint] = Self, predecessor: tuple[int, StreamEndpoint] = None):
        """
        Creates a new Chord Peer.

        :param node_id: Name of endpoint for logging purposes.
        :param name: Name of Peer.
        :param addr: Address of Peer.
        :param finger_table:
        :param successor: Successor of node on Chord Ring, initialized as self
        :param predecessor: Predecessor of node on Chord Ring, intialized as None
        """

        self._id = node_id
        self._name = name
        self._addr = addr
        self._finger_table = finger_table  # starts at 1
        self._successor = successor  # init to None, set at join to chord ring
        self._predecessor = predecessor

    def open_chord(self):
        # open accepting endpoint for joining nodes
        endpoint = StreamEndpoint(name=f"Acceptor-Public-{Chordnode._name}", addr=self._addr, acceptor=True,
                                  multithreading=True, buffer_size=10000)
        endpoint.start()
        while True:
            try:
                print(f"{self._name}-{endpoint.receive(5)}")
            except TimeoutError:
                print(f"{self._name}-Public Endpoint opened, pls join!")

    def join_chord(self, remote_addr: tuple[str, int]):
        # send join request
        endpoint = StreamEndpoint(name=f"Acceptor-Public-{Chordnode._name}", addr=self._addr,
                                  remote_addr=self._remote_addr, acceptor=True, multithreading=True, buffer_size=10000)
        endpoint.start()
        endpoint.send(Chordmessage(message_type="join", sender_id=self._id, sender_addr=self._addr))
        # TODO implement join protocol

    def fix_fingertable(self):
        """
        Function to keep fingertables current with nodes joining and leaving the chordring. Should be called
        periodically by each node.
        """
        for i in range(self._finger_table.len()):
            self.find_successor(self._remote_addr, self._id + 2 ** i)

    def find_successor(self, sender_addr: tuple[str, int], find_succ_id: int):
        """
        Function to find successor(find_succ_id). If current node does no know successor(find_succ_id) it
        will ask its successor. If successor(find_succ_id) is found, its id and addr will be sent back directly to the
        node who initially send the request.
        :param sender_addr:
        :param find_succ_id:
        """
        # sender addr should be remote address of node who initially asked for succ of id
        if self._id < find_succ_id <= self._successor[0]:  # if i know succ of id -> send succ back to asking node
            # add successor info to chordmessage
            succ_found_msg = Chordmessage(message_type="find_successor_success", sender_id=self._id,
                                          sender_addr=self._addr, payload=self._successor)
            # check whether find_succ_id is in fingertable or open new endpoint
            if self._finger_table.get(find_succ_id):
                self._finger_table.get(find_succ_id).send(succ_found_msg)
            else:
                endpoint = StreamEndpoint(name=f"Sender-find_successor-id:{find_succ_id}", addr=self._addr,
                                          remote_addr=sender_addr, acceptor=False, multithreading=True,
                                          buffer_size=10000)
                endpoint.start()
                endpoint.send(succ_found_msg)
                endpoint.stop(shutdown=True)  # ?
        else:  # if IDK -> ask node that closest preceedes find_succ_id
            for finger in self._finger_table.keys():
                if self._id < self._finger_table.get(finger) < find_succ_id:
                    self._finger_table.get(finger).send(
                        Chordmessage(message_type="find_successor", reply_addr=sender_addr, find_id=find_succ_id))
                    break


class Chordmessage:
    """
    Class for Chord messages
    """
    # TODO unfinished
    # FIXME address situation
    _message_type: str
    _sender_id: int
    _sender_addr: tuple[str, int]
    _reply_addr: tuple[str, int]
    _find_id: int
    _find_remote_addr: tuple[str, int]
    _payload: object

    def __init__(self, message_type: str = "unspecified", sender_id: int = -1, sender_addr: tuple[str, int] = None,
                 reply_addr: tuple[str, int] = None, find_id: int = -1, find_remote_addr: tuple[str, int] = None,
                 payload: object = None):
        self._message_type = message_type
        self._sender_id = sender_id
        self._sender_addr = sender_addr
        self._reply_addr = reply_addr
        self._find_id = find_id
        self._find_remote_addr = find_remote_addr
        self._payload = payload


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    pass
