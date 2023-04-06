import pickle
import socket

import pyshark


def send_data(data):  # FIXME socket comm
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # client_socket.settimeout(1.0)
    addr = ("127.0.0.1", 12000)
    client_socket.connect(addr)
    already_sent = 0
    while already_sent < len(data):
        already_sent += client_socket.send(data[already_sent:])
    client_socket.close()


def pyshark_capture():
    capture = pyshark.LiveCapture(interface='eth0')  # FIXME add exception for socket communication
    for p in capture.sniff_continuously():
        send_data(pickle.dumps(p))


if __name__ == '__main__':
    pyshark_capture()
