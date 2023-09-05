import logging
import random
import threading
from time import sleep

from src.communication import StreamEndpoint


def threaded_acceptor(t_id: int):
    endpoint = StreamEndpoint(name=f"Acceptor-{t_id}", addr=("127.0.0.1", 32000 + t_id),
                              remote_addr=("127.0.0.1", 13000 + t_id),
                              acceptor=True, multithreading=True, buffer_size=10000)
    endpoint.start()

    i = 0
    while True:
        endpoint.send(f"{t_id}-pong {i}")
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

                endpoint = StreamEndpoint(name=f"Acceptor-{t_id}", addr=("127.0.0.1", 32000 + t_id),
                                          remote_addr=("127.0.0.1", 13000 + t_id),
                                          acceptor=True, multithreading=True, buffer_size=10000)
                endpoint.start()
            else:
                endpoint.stop()
                sleep(random.randrange(3))
                endpoint.start()


def multithreaded_acceptor(num_threads: int):
    for i in range(num_threads):
        threading.Thread(target=threaded_acceptor, args=(i,)).start()
        sleep(random.randrange(2))


def clashing_acceptor():
    endpoint_1 = StreamEndpoint(name=f"Acceptor-{1}", addr=("127.0.0.1", 13000),
                                remote_addr=("127.0.0.1", 32000),
                                acceptor=True, multithreading=True, buffer_size=10000)

    endpoint_2 = StreamEndpoint(name=f"Acceptor-{2}", addr=("127.0.0.1", 13000),
                                remote_addr=("127.0.0.1", 32000),
                                acceptor=True, multithreading=True, buffer_size=10000)


def single_message_acceptor():
    endpoint = StreamEndpoint(name="Acceptor", addr=("127.0.0.1", 32000), remote_addr=("127.0.0.1", 13000),
                              acceptor=True, multithreading=True, buffer_size=10000)
    endpoint.start()

    print(endpoint.receive())
    endpoint.stop()
    print("No Block")


def simple_acceptor():
    endpoint = StreamEndpoint(name="Acceptor", addr=("127.0.0.1", 32000), remote_addr=("127.0.0.1", 13000),
                              acceptor=True, multithreading=True, buffer_size=10000)
    endpoint.start()

    i = 0
    while True:
        endpoint.send(f"pong {i}")
        try:
            print(endpoint.receive(5))
        except TimeoutError:
            print("nothing to receive")
        sleep(2)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)

    # simple_acceptor()
    # single_message_acceptor()
    multithreaded_acceptor(100)
    # clashing_acceptor()
