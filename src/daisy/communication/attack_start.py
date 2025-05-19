"""

Author: Sandra Schneider
Modified: 15.05.2025
"""

import logging
import random
import threading
from time import sleep

from daisy.communication import StreamEndpoint

attack_info = []

def single_message_acceptor():
    """Creates and starts an acceptor to perform a single receive before stopping the
    endpoint, to test if endpoints can be stopped while they are receiving multiple
    messages.
    """
    endpoint = StreamEndpoint(
        name="Acceptor",
        addr=("127.0.0.1", 32000), #???
        remote_addr=("127.0.0.1", 13000),#???
        acceptor=True,
        multithreading=True,
        buffer_size=10000,
    )
    endpoint.start()
      
    try:
        msg = endpoint.receive(5)
    except TimeoutError:
        print("nothing to receive")
    sleep(2)

    endpoint.stop

    message_to_list(msg)



def message_to_list(msg):
    global attack_info 
    attack_info = msg.split("§",5)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )