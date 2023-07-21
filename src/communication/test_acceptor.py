import logging
import threading
from time import sleep
import src.communication.message_stream as stream


def t1():
    dummy.start()
#
#     while True:
#         dummy.send("pong")
#         try:
#             print(dummy.receive(1))
#         except TimeoutError:
#             print("nothing to receive")


def t2():
    endpoint.start()

    while True:
        endpoint.send("pong")
        try:
            print(endpoint.receive(5))
        except TimeoutError:
            print("nothing to receive")
        sleep(2)
    # for x in endpoint:
    #     print(x)


logging.basicConfig(format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.DEBUG)

dummy = stream.StreamEndpoint(name="Dummy", addr=("127.0.0.1", 32000), remote_addr=("127.0.0.1", 15000),
                              acceptor=True, multithreading=True, buffer_size=10000)
threading.Thread(target=t1, daemon=True).start()

endpoint = stream.StreamEndpoint(name="Receiver", addr=("127.0.0.1", 32000), remote_addr=("127.0.0.1", 13000),
                                 acceptor=True, multithreading=True, buffer_size=10000)
threading.Thread(target=t2, daemon=True).start()
