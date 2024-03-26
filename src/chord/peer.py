import argparse
import random
import threading
import typing
from time import sleep, time
from typing import Optional, List, Any
from uuid import uuid4
from enum import Enum

from daisy.communication import EndpointServer, StreamEndpoint


def close_tmp_ep(ep: StreamEndpoint, sleep_time: int = 10, stop_timeout: int = 10):
    sleep(sleep_time)
    ep.stop(shutdown=True, timeout=stop_timeout)


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
    """Class for Chord messages.
    """
    id: uuid4
    type: MessageType
    peer_tuple: tuple[int, tuple[str, int]]
    origin: MessageOrigin

    def __init__(self, message_id: uuid4, type: MessageType, peer_tuple: tuple[int, tuple[str, int]] = None,
                 origin: MessageOrigin = None):
        """Creates a new Chordmessage.
        :param message_id: Message identifier
        :param type: Type of message for processing in receive function.
        :param peer_tuple: Id and address of the peer sent whithin the Chordmessage.
        """
        self.id = message_id
        self.type = type
        self.peer_tuple = peer_tuple
        self.origin = origin


def send(send_addr: tuple[str, int], message: Chordmessage):
    endpoint = StreamEndpoint(name=f"join", remote_addr=send_addr,
                              acceptor=False, multithreading=True, buffer_size=10000)
    endpoint.start()
    endpoint.send(message)
    threading.Thread(target=lambda: close_tmp_ep(endpoint, 10, 10), daemon=True).start()


class Peer:
    _id: int
    _addr: tuple[str, int]

    _successor: tuple[int, tuple[str, int]] | None
    _successor_endpoint: StreamEndpoint | None
    _predecessor: tuple[int, tuple[str, int]] | None
    _predecessor_endpoint: StreamEndpoint | None

    _endpoint_server: EndpointServer

    def __init__(self, p_id: int, addr: tuple[str, int], successor: tuple[int, tuple[str, int]],
                 predecessor: tuple[int, tuple[str, int]]):
        self._id = p_id
        self._addr = addr
        self._successor = successor
        self._successor_endpoint = None
        self._predecessor = predecessor
        self._predecessor_endpoint = None
        self._endpoint_server = EndpointServer(f"{id}-EpS", addr=addr, multithreading=True, c_timeout=30)

    def _create(self):
        self._endpoint_server.start()
        # self._successor = (self._id, self._addr)
        self._predecessor_endpoint = StreamEndpoint(name=f"pred-ep-{self._id}", remote_addr=self._predecessor[1],
                                                    acceptor=False, multithreading=True,
                                                    buffer_size=10000)
        self._predecessor_endpoint.start()
        self._successor_endpoint = StreamEndpoint(name=f"succ-ep-{self._id}", remote_addr=self._successor[1],
                                                  acceptor=False, multithreading=True,
                                                  buffer_size=10000)
        self._successor_endpoint.start()

    def _join(self, join_addr: tuple[str, int]):
        self._endpoint_server.start()
        message = Chordmessage(type=MessageType.JOIN, peer_tuple=(self._id, self._addr), message_id=1)
        send(join_addr, message)

    def _check_is_predecessor(self, check_id: int) -> bool:
        # check_id in [pred, n[
        pred_id = self._predecessor[0]
        return ((self._id < pred_id) and (check_id not in range(self._id + 1, pred_id + 1))) or (
                check_id in range(pred_id + 1, self._id + 1))

    def _check_is_successor(self, check_id: int) -> bool:
        # check_id in ]n, succ]
        succ_id = self._successor[0]
        return ((self._id > succ_id) and (check_id not in range(succ_id + 1, self._id + 1))) or (
                check_id in range(self._id + 1, succ_id + 1))

    def _try_to_find_lookup_result_locally(self, lookup_id) -> tuple[int, tuple[str, int]] | None:
        if lookup_id == self._predecessor[0]:
            return self._predecessor
        elif lookup_id == self._successor[0] or self._check_is_successor(lookup_id):
            return self._successor
        elif self._check_is_predecessor(lookup_id):
            return self._id, self._addr
        else:
            return None

    def _lookup(self, lookup_id: int, response_addr: tuple[str, int]):
        result = self._try_to_find_lookup_result_locally(lookup_id)
        print(result)
        if result is None:  # relay request to successor
            print(self._successor_endpoint.poll())
            self._successor_endpoint.send(
                Chordmessage(type=MessageType.LOOKUP_REQ, peer_tuple=(lookup_id, response_addr), message_id=1))
            return
        message = Chordmessage(type=MessageType.LOOKUP_RES, peer_tuple=result, message_id=1)
        send(response_addr, message)

    def _get_r_ready_eps(self) -> set[StreamEndpoint]:
        r_ready_eps = set()
        if self._successor_endpoint and self._successor_endpoint.poll()[0][1]:
            r_ready_eps.add(self._successor_endpoint)
        if self._predecessor_endpoint and self._predecessor_endpoint.poll()[0][1]:
            r_ready_eps.add(self._predecessor_endpoint)
        # get r_ready eps from ep server
        r_ready, _ = self._endpoint_server.poll_connections()
        for addr in r_ready:
            r_ready_eps.add(r_ready[addr])
        return r_ready_eps

    def run(self, join_addr: tuple[str, int] = None):
        if join_addr is None:
            self._create()
        else:
            self._join(join_addr=join_addr)

        while True:
            r_ready = self._get_r_ready_eps()
            for ep in r_ready:
                message = typing.cast(Chordmessage, ep.receive(timeout=0))
                print(message.peer_tuple, message.origin, message.type)
                match message.type:
                    case MessageType.LOOKUP_REQ:
                        print(f"starting lookup for {message.peer_tuple}")
                        self._lookup(*message.peer_tuple)
                    case MessageType.LOOKUP_RES:
                        print(f"Received lookup result {message.peer_tuple}")
                        self._successor = message.peer_tuple
                        self._successor_endpoint = StreamEndpoint(name=f"succ-ep-{self._id}",
                                                                  remote_addr=self._successor[1],
                                                                  acceptor=False, multithreading=True,
                                                                  buffer_size=10000)
                        print(self._successor)
                    case MessageType.JOIN:
                        print(f"sending join lookup for {message.peer_tuple}")
                        self._lookup(*message.peer_tuple)

    def get_id(self): # only for testing
        return self._id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--id", type=int, default=None, help="Id of peer")
    parser.add_argument("--port", type=int, default=None, help="Port of peer")
    parser.add_argument("--succId", type=int, default=None, help="successor id")
    parser.add_argument("--succPort", type=int, default=None, help="successor port")
    parser.add_argument("--predId", type=int, default=None, help="predecessor tuple")
    parser.add_argument("--predPort", type=int, default=None, help="predecessor port")

    args = parser.parse_args()

    localhost = "127.0.0.1"

    peer = Peer(p_id=args.id, addr=(localhost, args.port), successor=(args.succId, (localhost, args.succPort)),
                predecessor=(args.predId, (localhost, args.predPort)))
    if peer.get_id() == 2:
        peer.run((localhost, 10050))  # start as first chord peer
    else:
        peer.run()  # join existing chord
