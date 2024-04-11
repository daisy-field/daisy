# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Simple addition to the Acceptor/Initiator ping-pong tests, that creates a server
to regularly polls its (acceptor) endpoint connections to receive "pings" and respond
with "pongs", to test the behavior of handling concurrent connections.

Author: Fabian Hofmann
Modified: 10.04.24
"""

import logging
from time import sleep

from daisy.communication import EndpointServer


def simple_server():
    """Setup and start of ping-pong server (see module docstring)."""
    with EndpointServer(
        name="Testserver", addr=("127.0.0.1", 13000), c_timeout=60, multithreading=True
    ) as server:
        i = 0
        while True:
            r, w = server.poll_connections()
            for connection in r.items():
                try:
                    print(connection[1].receive(0))
                except TimeoutError:
                    print("rip timeout")
            for connection in w.items():
                connection[1].send(f"pong {i}")
            sleep(1)
            i += 1


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    simple_server()
