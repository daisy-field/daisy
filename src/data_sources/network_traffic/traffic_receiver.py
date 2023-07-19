import logging

import src.communication.message_stream as stream


def relay_pyshark_receiver():
    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.DEBUG)
    endpoint = stream.StreamEndpoint(addr=("127.0.0.1", 12000), acceptor=True,
                                     multithreading=True, buffer_size=10000)
    endpoint.start()

    for p in endpoint:
        print(p)


if __name__ == '__main__':
    relay_pyshark_receiver()
