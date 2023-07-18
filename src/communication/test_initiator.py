import logging
from time import sleep

import src.communication.message_stream as stream

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.DEBUG)
endpoint = stream.StreamEndpoint(addr=("127.0.0.1", 13000), remote_addr=("127.0.0.1", 12000), acceptor=False,
                                 multithreading=False, buffer_size=10000)
endpoint.start()

while True:
    endpoint.send("ping")
    # try:
    #     print(endpoint.receive(5))
    # except TimeoutError:
    #     print("nothing to receive")
    sleep(5)

