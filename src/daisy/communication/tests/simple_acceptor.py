# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Collection of various functions to test the acceptor-side (server) of the endpoint
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


def threaded_acceptor(t_id: int):
    """Creates and starts an acceptor with a specific ID to perform an endless ping-pong
    tests with the opposing initiator, sending out "pong" and receiving "ping" messages.
    In addition, at random intervals, stops or even shutdowns the endpoint to start or
    create it anew to test the resilience of the two endpoints.

    :param t_id: ID of thread.
    """
    endpoint = StreamEndpoint(
        name=f"Acceptor-{t_id}",
        addr=("127.0.0.1", 13000 + t_id),
        remote_addr=("127.0.0.1", 32000 + t_id),
        acceptor=True,
        multithreading=True,
        buffer_size=10000,
    )
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

        # if i % 10 == 0:
        #     if random.randrange(100) % 3 == 0:
        #         endpoint.stop(shutdown=True)
        #         sleep(random.randrange(3))
        #
        #         endpoint = StreamEndpoint(
        #             name=f"Acceptor-{t_id}",
        #             addr=("127.0.0.1", 32000 + t_id),
        #             remote_addr=("127.0.0.1", 13000 + t_id),
        #             acceptor=True,
        #             multithreading=True,
        #             buffer_size=10000,
        #         )
        #         endpoint.start()
        #     else:
        #         endpoint.stop()
        #         sleep(random.randrange(3))
        #         endpoint.start()
        sleep(1)
        i += 1


def multithreaded_acceptor(num_threads: int):
    """Starts n acceptor endpoints as separate threads, to test if endpoints can work
    in tandem using the shared underlying class attributes of the endpoint socket.

    :param num_threads: Number of acceptor threads to start.
    """
    for i in range(num_threads):
        threading.Thread(target=threaded_acceptor, args=(i,)).start()
        sleep(random.randrange(2))


def clashing_acceptor():
    """Creates multiple acceptor endpoints that have the same address (which is
    supported by the underlying endpoint sockets) but also the same remote (initiator)
    address which should result in a double registration causing an error.
    """
    endpoint_1 = StreamEndpoint(  # noqa: F841
        name=f"Acceptor-{1}",
        addr=("127.0.0.1", 13000),
        remote_addr=("127.0.0.1", 32000),
        acceptor=True,
        multithreading=True,
        buffer_size=10000,
    )

    endpoint_2 = StreamEndpoint(  # noqa: F841
        name=f"Acceptor-{2}",
        addr=("127.0.0.1", 13000),
        remote_addr=("127.0.0.1", 32000),
        acceptor=True,
        multithreading=True,
        buffer_size=10000,
    )


def single_message_acceptor():
    """Creates and starts an acceptor to perform a single receive before stopping the
    endpoint, to test if endpoints can be stopped while they are receiving multiple
    messages.
    """
    endpoint = StreamEndpoint(
        name="Acceptor",
        addr=("127.0.0.1", 32000),
        remote_addr=("127.0.0.1", 13000),
        acceptor=True,
        multithreading=True,
        buffer_size=10000,
    )
    endpoint.start()

    print(endpoint.receive())
    endpoint.stop()
    print("No Block")


def simple_acceptor():
    """Creates and starts an acceptor to perform an endless ping-pong tests with the
    opposing initiator, sending out "pong" and receiving "ping" messages.
    """
    endpoint = StreamEndpoint(
        name="Acceptor",
        addr=("127.0.0.1", 32000),
        remote_addr=("127.0.0.1", 13000),
        acceptor=True,
        multithreading=True,
        buffer_size=10000,
    )
    endpoint.start()

    i = 0
    while True:
        endpoint.send(f"pong {i}")
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
        level=logging.DEBUG,
    )

    # simple_acceptor()
    # single_message_acceptor()
    # multithreaded_acceptor(5)
    # clashing_acceptor()
