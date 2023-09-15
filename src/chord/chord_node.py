"""
    Class for Federated Peer that implements a fault-tolerant version of the Chord Protocol

    Author: Lotta Fejzula
    Modified: 15.09.23
"""
import logging
import random
import threading
from time import sleep
from typing import Optional, Self
# Bootstrap-problem :c
from enum import Enum

from src.communication.message_stream import StreamEndpoint


class Chordnode:
    """Class for creating nodes for the Chord DHT
    """

    _id: int

    _name: str
    _addr: tuple[str, int]
    _remote_addr: tuple[str, int]

    _finger_table: [{int, StreamEndpoint}]
    # int could be index relative to owning node object -> id+2^
    # Save endpoints for lookup in ft instead of address tuples
    _successor: tuple[int, StreamEndpoint]
    _predecessor: tuple[int, StreamEndpoint] # int is id of succ/pred

    def __init__(self, id: int, name: str, addr: tuple[str, int], remote_addr: tuple[str, int] = None,
                 finger_table: [{int, tuple[str, int]}] = None, successor: tuple[int, StreamEndpoint] = Self,
                 predecessor: tuple[int, StreamEndpoint] = None):
        """Creates a new Chord Peer.

        :param id: Name of endpoint for logging purposes.
        :param name: Name of Peer.
        :param addr: Address of Peer.
        :param remote_addr: Address of chord-external peer endpoint to be connected to in order to join an existing chord-ring (initiator) or allow another peer to join (acceptor). Mandatory in initiator mode (acceptor set to
        false), for acceptor mode this fixes the remote endpoint that is allowed to be connected to this endpoint.
        :param finger_table:
        :param successor: Successor of node on Chord Ring, initialized as self
        :param predecessor: Predecessor of node on Chord Ring, intialized as None
        """

        self._id = id
        self._name = name
        self._addr = addr
        self._remote_addr = remote_addr
        self._finger_table = finger_table  # starts at 1
        self._successor = successor  # init to None, set at join to chord ring
        self._predecessor = predecessor

    def open_chord(self):
        # open accepting endpoint for joining nodes
        endpoint = StreamEndpoint(name=f"Acceptor-Public-{Chordnode._name}", addr=self._addr,
                                  remote_addr=self._remote_addr,
                                  acceptor=True, multithreading=True, buffer_size=10000)
        endpoint.start()
        while True:
            try:
                print(f"{self._name}-{endpoint.receive(5)}")
            except TimeoutError:
                print(f"{self._name}-Public Endpoint opened, waiting for you!")

    def join_chord(self, remote_addr: tuple[str, int]):
        # send join request
        endpoint = StreamEndpoint(name=f"Acceptor-Public-{Chordnode._name}", addr=self._addr,
                                  remote_addr=self._remote_addr,
                                  acceptor=True, multithreading=True, buffer_size=10000)
        endpoint.start()
        endpoint.send(Chordmessage("join", self._id, self._addr))  # FixMe gelbe kringelchen
    def fix_fingertable(self):
        for i in range(self._finger_table.len()):
            self.find_successor(self._id + 2 ** i)

    def find_successor(self, sender_addr:tuple[str, int], find_succ_id: int):  # i ask recv if they know succ of sender
        # TODO implement receiving find succ request in receive message handler
        # TODO implement receive message handler
        #recv id should be in fingretable
        if self._id < find_succ_id <= self._successor[0]:
            endpoint = StreamEndpoint(name=f"Acceptor-Public-{Chordnode._name}", addr=self._addr,
                                      remote_addr=sender_addr,
                                      acceptor=False, multithreading=True, buffer_size=10000)
            endpoint.start()
            endpoint.send(Chordmessage("join", self._id, self._remote_addr))
            # send kann vllt nicht funktionieren, der andere muss irgendwie ja auch recv können,
            # einfach remote addr in Antwort mitsenden? Nochmal fabian fragen
        else:
            self._predecessor[1].send(Chordmessage("find_successor", find_id=find_succ_id))  # send find succ of sender request
        pass


class Chordmessage:
    """Class to specify which type of messages and what content can be sent between peers to fullfill the chord protocol
    """

    _message_type: str
    _sender_id: int
    _find_id: int
    _find_addr: tuple[str, int]
    _find_remote_addr: tuple[str, int]

    def __init__(self, message_type: str, sender_id: int, find_id: int, find_addr: tuple[str, int], find_remote_addr: tuple[str, int]):
        self._message_type = message_type  # how do I say "should be one of ...", maybe not necessary
        self._sender_id = sender_id
        self._find_id = find_id
        self._find_addr = find_addr
        self._find_remote_addr = find_remote_addr


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    pass
