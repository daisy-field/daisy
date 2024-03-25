import argparse
import threading
from time import sleep
from typing import Optional, List, Any
from uuid import uuid4
from enum import Enum

from communication import EndpointServer, StreamEndpoint


def close_tmp_ep(ep: StreamEndpoint, sleep_time: int = 10, stop_timeout: int = 10):
    sleep(sleep_time)
    ep.stop(shutdown=True, timeout=stop_timeout)


class Peer:
    _id: int
    addr: tuple[str, int]

    _successor: tuple[int, tuple[str, int]] | None
    _successor_endpoint: StreamEndpoint | None
    _predecessor: tuple[int, tuple[str, int]] | None
    _predecessor_endpoint: StreamEndpoint | None

    _endpoint_server: EndpointServer

    def __init__(self, id: int, name: str, addr: tuple[str, int], successor: tuple[int, tuple[str, int]],
                 predecessor: tuple[int, tuple[str, int]]):
        self._id = id
        self._name = name
        self.addr = addr
        self._successor = successor
        self._predecessor = predecessor
        self._endpoint_server = EndpointServer(f"{name}-EpS", addr=addr, multithreading=True, c_timeout=30)

    def create(self):
        self._endpoint_server.start()
        self._predecessor_endpoint = StreamEndpoint(name=f"pred-ep-{self._id}", remote_addr=self._predecessor[1],
                                                    acceptor=False, multithreading=True,
                                                    buffer_size=10000)
        self._predecessor_endpoint.start()

    def _ping(self, addr: tuple[str, int], relationship: str):
        ep_name = f"ping-ep-{self._id}"
        endpoint = StreamEndpoint(name=ep_name, remote_addr=addr, acceptor=False, multithreading=True,
                                  buffer_size=10000)
        endpoint.start()
        endpoint.send(f"ping {relationship}")
        threading.Thread(target=lambda: close_tmp_ep(endpoint, 10, 10), daemon=True).start()

    def get_r_ready_eps(self) -> list[StreamEndpoint]:
        readable_eps = []
        r_ready, _ = self._endpoint_server.poll_connections()
        for addr in r_ready:
            readable_eps.append(r_ready[addr])
        return readable_eps

    def run(self):
        self.create()
        self._ping(self._predecessor[1], "pred")

        while True:
            r_ready = self.get_r_ready_eps()
            for ep in r_ready:
                message = ep.receive(timeout=0)
                if message and message == "ping pred":
                    self._ping(self._successor[0], "succ")
                if message and message == "ping succ":
                    self._ping(self._predecessor[0], "pred")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--id", type=int, default=None, help="Id of peer")
    parser.add_argument("--name", help="Peer name")
    parser.add_argument("--port", type=int, default=None, help="Port of peer")
    parser.add_argument("--succ", type=tuple[int, tuple[str, int]], default=None, help="successor tuple")
    parser.add_argument("--pred", type=tuple[int, tuple[str, int]], default=None, help="predecessor tuple")
    args = parser.parse_args()

    peer_ip = "127.0.0.1"

    peer = Peer(id=args.id, name=args.name, addr=(peer_ip, args.port), successor=args.succ, predecessor=args.pred)
    peer.run()
