"""
    Class for Federated Peer that implements a fault-tolerant version of the Chord Protocol

    Author: Lotta Fejzula
    Modified: 05.09.23
"""
import logging
import random
import threading
from time import sleep

# Bootstrap-problem :c

from src.communication.message_stream import EndpointSocket, StreamEndpoint
from typing import Optional


class ChordPeer:
    """Class to create a single federated peer
    """

    _id: str
    _name: str
    _addr: tuple[str, int]
    _remote_addr: Optional[tuple[str, int]]

    _finger_table: [{str: str}]
    _successor: str
    _predecessor: str

    _accepting_endpoint_internal: EndpointSocket
    _accepting_endpoint_external: EndpointSocket
    _accepting_endpoint_external: EndpointSocket
    _initiating_endpoint_internal: EndpointSocket

    def __init__(self, id: str, name: str, addr: str, remote_addr: tuple[str, int] = None, finger_table:[{str: str}] = None, successor: str = None, predecessor: str = None):
        """Creates a new Chord Peer.

        :param id: Name of endpoint for logging purposes.
        :param name: Name of Peer.
        :param addr: Address of Peer.
        :param remote_addr: Address of chord-external peer endpoint to be connected to in order to join an existing chord-ring (initiator) or allow another peer to join (acceptor). Mandatory in initiator mode (acceptor set to
        false), for acceptor mode this fixes the remote endpoint that is allowed to be connected to this endpoint.
        """

        self._id = id
        self._name = name
        self._addr = addr
        self._remote_addr = remote_addr
        self._finger_table = finger_table
        self._successor = successor
        self._predecessor = predecessor

    def create_chord(self):
        raise NotImplementedError("classmethod create_chord of class chord_peer has not yet been implemented.")

def threaded_acceptor(t_id: int):
    accept_endpoint = StreamEndpoint(name=f"Acceptor-{t_id}", addr=("127.0.0.1", 32000 + t_id),
                              remote_addr=("127.0.0.1", 13000 + t_id),
                              acceptor=True, multithreading=True, buffer_size=10000)
    accept_endpoint.start()

    i = 0
    while True:
        accept_endpoint.send(f"{t_id}-pong {i}")
        i += 1
        try:
            print(f"{t_id}-{accept_endpoint.receive(random.randrange(5))}")
        except TimeoutError:
            print(f"{t_id}-" + "nothing to receive")
        # sleep(random.randrange(3))

        if i % 10 == 0:
            if random.randrange(100) % 3 == 0:
                accept_endpoint.stop(shutdown=True)
                sleep(random.randrange(3))

                accept_endpoint = StreamEndpoint(name=f"Acceptor-{t_id}", addr=("127.0.0.1", 32000 + t_id),
                                          remote_addr=("127.0.0.1", 13000 + t_id),
                                          acceptor=True, multithreading=True, buffer_size=10000)
                accept_endpoint.start()
            else:
                accept_endpoint.stop()
                sleep(random.randrange(3))
                accept_endpoint.start()



def threaded_initiator(t_id: int):
    endpoint = StreamEndpoint(name=f"Initiator-{t_id}", addr=("127.0.0.1", 13001 + t_id),
                              remote_addr=("127.0.0.1", 32001 + t_id),
                              acceptor=False, multithreading=True, buffer_size=10000)
    endpoint.start()

    i = 0
    while True:
        endpoint.send(f"{t_id}-ping {i}")
        i += 1
        try:
            print(f"{t_id}-{endpoint.receive(random.randrange(5))}")
        except TimeoutError:
            print(f"{t_id}-" + "nothing to receive")
        # sleep(random.randrange(3))

        if i % 10 == 0:
            if random.randrange(100) % 3 == 0:
                endpoint.stop(shutdown=True)
                sleep(random.randrange(3))

                endpoint = StreamEndpoint(name=f"Initiator-{t_id}", addr=("127.0.0.1", 13001 + t_id),
                                          remote_addr=("127.0.0.1", 32001 + t_id),
                                          acceptor=False, multithreading=True, buffer_size=10000)
                endpoint.start()
            else:
                endpoint.stop()
                sleep(random.randrange(3))
                endpoint.start()

def multithreaded_endpoints(num_threads: int):
    for i in range(num_threads):
        threading.Thread(target=threaded_initiator, args=(i,)).start()
        sleep(random.randrange(2))
        threading.Thread(target=threaded_acceptor, args=(i,)).start()
        sleep(random.randrange(2))


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    multithreaded_endpoints(1)

