import logging

import pyshark

from communication.message_stream import StreamEndpoint


def pyshark_capture():
    logging.basicConfig(level=logging.DEBUG)
    endpoint = StreamEndpoint(addr=("127.0.0.1", 13000), remote_addr=("127.0.0.1", 12000), multithreading=False)
    endpoint.start(StreamEndpoint.EndpointType.SOURCE)

    capture = pyshark.LiveCapture(interface='eth0')  # FIXME add exception for socket communication
    for p in capture.sniff_continuously():
        endpoint.send(p)


if __name__ == '__main__':
    pyshark_capture()
