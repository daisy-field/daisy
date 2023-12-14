import logging
import random
import threading
from time import sleep

from src.communication import StreamEndpoint


def threaded_initiator(t_id: int):
    endpoint = StreamEndpoint(name=f"Initiator-{t_id}", addr=("127.0.0.1", 32000 + t_id),
                              remote_addr=("127.0.0.1", 13000),
                              acceptor=False, multithreading=True, buffer_size=10000)
    endpoint.start()

    i = 0
    while True:
        endpoint.send(f"{t_id}-ping {i}")
        try:
            print(f"{t_id}-{endpoint.receive(random.randrange(5))}")
        except TimeoutError:
            print(f"{t_id}-" + "nothing to receive")
        # sleep(random.randrange(3))

        if i % 10 == 0:
            if random.randrange(100) % 3 == 0:
                logging.warning("Shutting Down")
                endpoint.stop(shutdown=True)
                sleep(random.randrange(3))

                endpoint = StreamEndpoint(name=f"Initiator-{t_id}", addr=("127.0.0.1", 32000 + t_id),
                                          remote_addr=("127.0.0.1", 13000),
                                          acceptor=False, multithreading=True, buffer_size=10000)
                endpoint.start()
            else:
                logging.warning("Stopping")
                endpoint.stop()
                sleep(random.randrange(3))
                endpoint.start()
        sleep(1)
        i += 1


def multithreaded_initiator(num_threads: int):
    for i in range(num_threads):
        threading.Thread(target=threaded_initiator, args=(i,)).start()
        sleep(random.randrange(2))


def single_message_initiator():
    endpoint = StreamEndpoint(name="Initiator", addr=("127.0.0.1", 13000), remote_addr=("127.0.0.1", 32000),
                              acceptor=False, multithreading=True, buffer_size=10000)
    endpoint.start()

    endpoint.send(f"ping")


def simple_initiator():
    endpoint = StreamEndpoint(name="Initiator", addr=("127.0.0.1", 13000), remote_addr=("127.0.0.1", 32000),
                              acceptor=False, multithreading=True, buffer_size=10000)
    endpoint.start()

    i = 0
    while True:
        endpoint.send(f"ping {i}")
        try:
            print(endpoint.receive(5))
        except TimeoutError:
            print("nothing to receive")
        sleep(2)
        i += 1


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)

    # simple_initiator()
    # single_message_initiator()
    multithreaded_initiator(5)
