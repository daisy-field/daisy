# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
An efficient, persistent, and stateless communications stream between two endpoints over
BSD sockets. Supports SSL and LZ4 compression.

    * StreamEndpoint - Core class of the communications framework.
    * EndpointServer - Helper class to group acceptor endpoints together under one
    common address.
        * ep_select() - Helper function to poll a list of endpoints whether
        something can be read from/written to.
        * receive_latest_ep_objs() - Helper function to receive the latest messages
        from a list of endpoints.

Author: Fabian Hofmann
Modified: 03.04.24
"""

__all__ = ["StreamEndpoint", "EndpointServer", "connect_to_target","arpspoof_run","start_webserver","path_traversal","slowloris_run", "reverse_shell_target"]

from .message_stream import EndpointServer
from .message_stream import StreamEndpoint
from .connect_to_target import connect_to_target
from .arp_spoof import run as arpspoof_run
from .app import start_webserver, path_traversal
from .slowloris import main as slowloris_run
from .reverse_shell_target import reverse_shell_target
