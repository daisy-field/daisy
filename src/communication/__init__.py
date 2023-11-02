"""
    An efficient, persistent, and stateless communications stream between two endpoints over BSD sockets. Supports SSL
    and LZ4 compression.

        * StreamEndpoint - Core class of the communications framework.
        * EndpointServer - Helper class to group acceptor endpoints together under one common address.
        * ep_select()    - Helper function to poll a list of endpoints whether something can be read from/written to.
    Author: Fabian Hofmann
    Modified: 30.10.23
"""

from .message_stream import StreamEndpoint
from .message_stream import EndpointServer
from .message_stream import ep_select
