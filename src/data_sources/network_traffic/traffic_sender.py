"""
    Small pyshark forwarding relay, should ONLY be used when capturing packets on a remote machine due to the overhead
    of sockets and system buffers.

    Author: Fabian Hofmann
    Modified: 13.04.23
"""

import logging

import pyshark

import src.communication.message_stream as stream


# TODO add args to main for deployment

def relay_pyshark_captures():
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.DEBUG)
    endpoint = stream.StreamEndpoint(addr=("127.0.0.1", 13000), remote_addr=("127.0.0.1", 12000), acceptor=False,
                                     multithreading=True, buffer_size=10000)
    endpoint.start()

    capture = pyshark.LiveCapture(interface='enx0826ae3a9e6e')  # FIXME add exception for endpoint communication
    for p in capture.sniff_continuously():
        endpoint.send(p)


if __name__ == '__main__':
    relay_pyshark_captures()
