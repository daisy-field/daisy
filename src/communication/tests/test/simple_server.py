import logging
from time import sleep

from src.communication import EndpointServer


def simple_server():
    with EndpointServer(name="Testserver", addr=("127.0.0.1", 13000), c_timeout=5, multithreading=True) as server:
        while True:
            r, w = server.poll_connections()
            for connection in r.items():
                print(connection[1].receive(0))
            for connection in w.items():
                connection[1].send(f"pong")
            sleep(10)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    simple_server()
