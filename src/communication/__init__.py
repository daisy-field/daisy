"""
    An efficient, persistent, and stateless communications stream between two endpoints over BSD sockets. Supports SSL
    and LZ4 compression.

        * StreamEndpoint - Core class of the communications framework.

    Author: Fabian Hofmann
    Modified: 04.08.23
"""

from .message_stream import StreamEndpoint
