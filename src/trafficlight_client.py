"""
    Quick testing client to check the functionality of the tl-service. The ROS-wrapper has to do this only, everything
    else is already taken care of via the (ROS-less) service itself.

    Author: Fabian Hofmann
    Modified: 21.9.22
"""

import select
import socket

if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(b"{\"tl_id\": \"10000000000000063\"}", ("0.0.0.0", 8800))

    sock.setblocking(False)
    ready = select.select([sock], [], [], 0.01)
    if ready[0]:
        data, _ = sock.recvfrom(8 * 1024)
        print(data.decode())
    else:
        print("TL Service Offline!")
