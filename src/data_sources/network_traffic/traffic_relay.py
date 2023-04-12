import logging

import pyshark

import communication.message_stream as stream


def pyshark_capture():
    logging.basicConfig(level=logging.DEBUG)
    endpoint = stream.StreamEndpoint(addr=("127.0.0.1", 13000), remote_addr=("127.0.0.1", 12000),
                                     endpoint_type=stream.SOURCE, multithreading=True, buffer_size=10000)
    endpoint.start()

    capture = pyshark.LiveCapture(interface='wlp3s0')  # FIXME add exception for socket communication
    for p in capture.sniff_continuously():
        endpoint.send(p)


if __name__ == '__main__':
    pyshark_capture()
