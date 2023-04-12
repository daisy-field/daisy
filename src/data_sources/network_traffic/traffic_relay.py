import logging

import pyshark

import communication.message_stream as stream


def pyshark_capture():
    logging.basicConfig(level=logging.DEBUG)
    endpoint = stream.StreamEndpoint(addr=("127.0.0.1", 13000), remote_addr=("127.0.0.1", 12000),
                                     endpoint_type=stream.SOURCE, multithreading=False)
    endpoint.start()

    capture = pyshark.LiveCapture(interface='enx0826ae3a9e6e')  # FIXME add exception for socket communication
    for p in capture.sniff_continuously():
        endpoint.send(p)


if __name__ == '__main__':
    pyshark_capture()
