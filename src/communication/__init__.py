"""
    An efficient, persistent, and stateless communications stream between two endpoints over BSD sockets. Supports SSL
    and LZ4 compression.

        * StreamEndpoint - Core class of the communications framework.
        * EndpointServer -

    Author: Fabian Hofmann
    Modified: 30.10.23
"""

from .message_stream import StreamEndpoint
from .message_stream import EndpointServer
