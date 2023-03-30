"""
    ROS-less stand-alone version of the traffic light service in BeIntelli. Basically this is just a wrapper around
    the trafficlight_receiver that handles the processing of the traffic light messages from the Cohda box, while also
    being "wrapped" by the ROS-service. Necessary due to the absence of the DSRCpy library in the ROS docker setup,
    in addition to making the configuration of thereof easier.

    Author: Fabian Hofmann
    Modified: 21.9.22
"""
import argparse
import json
import logging
import pathlib
import socket
from threading import Thread

from trafficlight_receiver import trafficlight_receiver as tl


def listen_to_tl_requests(ip, port):
    """Handles the request to (known) traffic light states from the cohda boxes on the road via a simple JSON- and UDP-
    based request-reply exchange between client and service. Clients request the information on a traffic light via the
    ID it has on the hd-map, the service converts this id to the one used by the cohda boxes and looks it up, responding
    accordingly to the client's request.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((ip, port))

    while True:
        logging.debug("Waiting for packets")
        data, addr = sock.recvfrom(8 * 1024)
        logging.debug(f"Received packet from {addr}")

        request = json.loads(data)
        tl_id = request["tl_id"]

        if tl_id in tl_mapping and tl_mapping[tl_id] in tl.traffic_light_states:
            response = json.dumps(tl.traffic_light_states[tl_mapping[tl_id]])
        else:
            response = f"{{ \"Error\": \"ID does not exist in mapping ({tl_id})!\"}}"

        sock.sendto(response.encode(), addr)
        logging.debug(f"Processed request: {json.dumps(request)}\n\tres: {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--debug", type=bool, default=False, help="Show debug outputs")
    parser.add_argument("--RXip", default="0.0.0.0",
                        help="IP of traffic light service")
    parser.add_argument("--RXport", type=int, default=8800,
                        help="UDP port of traffic light service")
    parser.add_argument("--mappingPath", type=pathlib.Path, default="mapping.json",
                        help="Path to mapping file (HD-Map TL-ID -> DSRC TL-ID)")
    parser.add_argument("--demo", type=bool, default=False, help="If enabled, use sample SPATEMs instead")
    parser.add_argument("--demoPath", type=pathlib.Path,
                        default="trafficlight_receiver/its/sample_spatems",
                        help="If demo set, path to directory with sample SPATEMs as jsons")
    parser.add_argument("--dsrcRXip", default="0.0.0.0",
                        help="IP to which the Cohda unit sends its messages")
    parser.add_argument("--dsrcRXport", type=int, default=4400,
                        help="UDP port to which the Cohda unit sends its messages")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    # either sample messages or cohda communication as traffic light state source
    if args.demo:
        logging.info("Demo detected. Processing a few collected DSRC packets for testing purposes...")
        tl.start_demo(args.demoPath)
    else:
        logging.info("Starting DSRC listening server...")
        Thread(target=tl.listen_to_dsrc_messages, args=(args.dsrcRXip, args.dsrcRXport)).start()

    # retrieval of the (current) mapping from hd-map TL ids to the DSRC TL ids used in this backend's datastructure
    with open(args.mappingPath) as f:
        tl_mapping = json.load(f)

    logging.info("Starting TL-service listening server...")
    listen_to_tl_requests(args.RXip, args.RXport)
