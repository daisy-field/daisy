import logging
import threading

import src.communication.message_stream as stream


def t1():
    dummy.start()


def t2():
    endpoint.start()

    while True:
        endpoint.send("pong")
        try:
            print(endpoint.receive())
        except TimeoutError:
            print("nothing to receive")


logging.basicConfig(format="%(asctime)s %(levelname)-5s %(name)-10s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.DEBUG)

dummy = stream.StreamEndpoint(name="Dummy", addr=("127.0.0.1", 31000), remote_addr=("127.0.0.1", 14000),
                              acceptor=True, multithreading=True, buffer_size=10000)
endpoint = stream.StreamEndpoint(name="Receiver", addr=("127.0.0.1", 32000), remote_addr=("127.0.0.1", 13000),
                                 acceptor=True, multithreading=True, buffer_size=10000)

threading.Thread(target=t1, daemon=True).start()
threading.Thread(target=t2, daemon=True).start()
