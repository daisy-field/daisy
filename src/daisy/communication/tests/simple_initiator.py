# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Collection of various functions to test the initiator-side (client) of the endpoint
class. These test-functions can be called directly, with the main on the bottom
adjusted for each test case.

Author: Fabian Hofmann
Modified: 10.04.24

"""

import logging
import random
import threading
from time import sleep

from daisy.communication import StreamEndpoint


def threaded_initiator(t_id: int):
    """Creates and starts an initiator with a specific ID to perform an endless
    ping-pong tests with the opposing acceptor, sending out "ping" and receiving
    "pong" messages. In addition, at random intervals, stops or even shutdowns the
    endpoint to start or create it anew to test the resilience of the two endpoints.

    :param t_id: ID of thread.
    """
    endpoint = StreamEndpoint(
        name=f"Initiator-{t_id}",
        addr=("127.0.0.1", 32000 + t_id),
        remote_addr=("127.0.0.1", 13000),
        # remote_addr=("127.0.0.1", 13000 + t_id),
        acceptor=False,
        multithreading=True,
        buffer_size=10000,
    )
    endpoint.start()

    i = 0
    while True:
        endpoint.send(f"{t_id}-ping {i}")
        try:
            print(f"{t_id}-{endpoint.receive(random.randrange(5))}")
        except TimeoutError:
            print(f"{t_id}-" + "nothing to receive")
        # sleep(random.randrange(3))

        # if i % 10 == 0:
        #     if random.randrange(100) % 3 == 0:
        #         logging.warning("Shutting Down")
        #         endpoint.stop(shutdown=True)
        #         sleep(random.randrange(3))
        #
        #         endpoint = StreamEndpoint(
        #             name=f"Initiator-{t_id}",
        #             addr=("127.0.0.1", 32000 + t_id),
        #             remote_addr=("127.0.0.1", 13000),
        #             acceptor=False,
        #             multithreading=True,
        #             buffer_size=10000,
        #         )
        #         endpoint.start()
        #     else:
        #         logging.warning("Stopping")
        #         endpoint.stop()
        #         sleep(random.randrange(3))
        #         endpoint.start()
        sleep(1)
        i += 1


def multithreaded_initiator(num_threads: int):
    """Starts n initiator endpoints as separate threads, to test if endpoints can work
    in tandem using the shared underlying class attributes of the endpoint socket.

    :param num_threads: Number of acceptor threads to start.
    """
    for i in range(num_threads):
        threading.Thread(target=threaded_initiator, args=(i,)).start()
        sleep(random.randrange(2))


def single_message_initiator():
    """Creates and starts an initiator to perform a single send before stopping the
    endpoint, to test if endpoints can be stopped while they are about to send a
    message. This test is not deterministic due to scheduling.
    """
    endpoint = StreamEndpoint(
        name="Initiator",
        addr=("127.0.0.1", 13000),
        remote_addr=("127.0.0.1", 32000),
        acceptor=False,
        multithreading=True,
        buffer_size=10000,
    )
    endpoint.start()

    endpoint.send("ping")

    endpoint.stop()


def simple_initiator():
    """Creates and starts an initiator to perform an endless ping-pong tests with the
    opposing initiator, sending out "ping" and receiving "pong" messages.
    """
    endpoint = StreamEndpoint(
        name="Initiator",
        addr=("127.0.0.1", 13000),
        remote_addr=("127.0.0.1", 32000),
        acceptor=False,
        multithreading=True,
        buffer_size=10000,
    )
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
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # simple_initiator()
    # single_message_initiator()
    multithreaded_initiator(5)
