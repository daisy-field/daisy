"""
    ROS-less stand-alone version of the v2x endpoint for communication with the traffic lights in BeIntelli. Receives
    UDP packets from Cohda box and parses them to json using the DSRCpy library. Any received SPAT messages (TL info)
    are processed and traffic light information is held in a datastructure to allow further use of the information.

    Code is based on Diginet ITS code for the vehicle by Martin Berger (18.6.21)

    Author: Fabian Hofmann, Martin Berger
    Modified: 21.9.22
"""

import argparse
import json
import logging
import os
import pathlib
import socket
import time

from dsrcpy import ETS

# lazy solution so this piece of code can be both be executed via main and called through package
try:
    from .its import spat
except ImportError:
    from its import spat

# lists all known traffic lights per intersection and last recorded color + interval
traffic_light_states = {}


def handle_dsrc_json(packet):
    packet['__received'] = time.time()

    if 'SPATEM' in packet:
        updated_states = spat.handle_spatem(packet)
        traffic_light_states.update(updated_states)


def listen_to_dsrc_messages(ip, port):
    """If the connected Cohda box is running the ETSI send-receive code (see readme) and is configured to forward it all
    to an external address, the host running this procedure is able ot receive these UDP packages and process the DSRC
    messages. Note that all packets containing messages other than SPAT (traffic light info) are discarded.
    """

    # noinspection PyShadowingNames
    def parse_btp_packet(packet):
        result, _ = ets.parse_btp_packet(packet)  # new DSRCpy version returns json along payload_len
        if result is not None:
            result = json.loads(result)
        return result

    ets = ETS()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((ip, port))

    while True:
        logging.debug("Waiting for packets")
        data, _ = sock.recvfrom(8 * 1024)
        logging.debug("Received packet")

        packet = parse_btp_packet(data)
        if packet is None:
            logging.warning("Could not parse packet:")
            logging.warning(data)
        else:
            try:
                handle_dsrc_json(packet)
            except ValueError as e:
                logging.exception(f"ERROR PROCESSING DSRC-PACKET: {e}")


def start_demo(path):
    """Quickly fill the traffic light states with information all intersection's traffic light information, using old
    collected messages (week of Aug 22nd-26th (2022)).

    @note Sample SPATEMs generate warnings at runtime because they are in the past (duh)
    @note Timestamps of events should not be used because depending on version, receive time stamps are set at runtime!
    """
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            with open(file_path) as f:
                spatem = json.load(f)
                handle_dsrc_json(spatem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--debug", type=bool, default=False, help="Show debug outputs")
    parser.add_argument("--demo", type=bool, default=False, help="If enabled, use sample SPATEMs instead")
    parser.add_argument("--demoPath", type=pathlib.Path, default="its/sample_spatems",
                        help="If demo set, path to directory with sample SPATEMs as jsons")
    parser.add_argument("--RXip", default="0.0.0.0", help="IP to which the Cohda unit sends its messages")
    parser.add_argument("--RXport", type=int, default=4400, help="UDP port to which the Cohda unit sends its messages")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    if args.demo:
        logging.info("Demo detected. Processing a few collected packets for testing purposes...")
        start_demo(args.demoPath)
    else:
        logging.info("Starting DSRC listening server...")
        listen_to_dsrc_messages(args.RXip, args.RXport)
