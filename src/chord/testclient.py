import threading
import typing
from time import time, sleep
from uuid import uuid4

from chord.chord_peer import MessageType, close_tmp_ep
from chord.lib import Chordmessage
from communication import EndpointServer, StreamEndpoint


class TestPeer:
    # TODO: have succ in finger table and not external
    """Class for Chordpeers.
    """

    _id: int
    _name: str  # for debugging and fun
    _addr: tuple[str, int]
    _endpoint_server: EndpointServer

    def __init__(self, name: str, addr: tuple[str, int], id_test: int = None):
        """
        Creates a new Chord Peer.

        :param name: Name of peer for logging and fun.
        :param addr: Address of peer.
        :param max_fingers: Max size of fingertable, also used for calculating peer ids on chord
        """

        self._id = id_test  # hash(addr) % (2 ** max_fingers)
        self._name = name
        self._addr = addr
        self._endpoint_server = EndpointServer(f"{name}-endpointserver", addr=addr, multithreading=True, c_timeout=30)

    def send_lookup(self, remote_addr: tuple[str, int]):
        id = typing.cast(int, input("enter lookup value"))
        message = Chordmessage(message_id=uuid4(), message_type=MessageType.LOOKUP_SUCC_REQ,
                               peer_tuple=(id, self._addr))
        ep_name = f"ep-{id}"
        endpoint = StreamEndpoint(name=ep_name, remote_addr=remote_addr, acceptor=False, multithreading=True,
                                  buffer_size=10000)
        endpoint.start()
        endpoint.send(message)

        threading.Thread(target=lambda: close_tmp_ep(endpoint, 10, 10), daemon=True).start()
        sleep(5)
        self.receive(id)

    def receive(self, id: int):
        now = time()
        while True:
            r_ready, _ = self._endpoint_server.poll_connections()
            for adr in r_ready:
                ep = r_ready[adr]
                recv = typing.cast(Chordmessage, ep.receive(timeout=0)).peer_tuple
                if recv is not None:
                    print(recv)
                    return
                sleep(2)

    def start(self):
        self._endpoint_server.start()


if __name__ == '__main__':
    testpeer = TestPeer("localhost", ("127.0.0.1", 15888), id_test=1)
    testpeer.start()
    addr = input("enter target port: ")
    testpeer.send_lookup(("127.0.0.1", typing.cast(int, addr)))
    while True:
        cmd = input("enter command:")
        match cmd:
            case "send":
                addr = input("enter target port: ")
                testpeer.send_lookup(("127.0.0.1", typing.cast(int, addr)))
            case "exit":
                break
