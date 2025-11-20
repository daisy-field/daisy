"""send messages to coordinate attacks and their recording.

Author: Sandra Schneider
Modified: 20.11.2025
"""

import logging

import argparse
import json
import logging

import argparse
import ipaddress
from zoneinfo import ZoneInfo
from time import sleep
from datetime import datetime, timezone, timedelta

from daisy.communication import StreamEndpoint


def initiat_attack_massages(ip, attack_massage):
    """
    Sends an attack definition message to a remote machine.
    
    Parameters:
        ip (str): IP address of the target host, e.g., "127.0.0.1".
        attack_massage (bytes): Serialized attack description (JSON encoded as bytes).
    """
     # Create a TCP client endpoint that connects to the remote system
    endpoint = StreamEndpoint(
        name="attack_massage",
        remote_addr=(ip, 32000), # Remote IP and TCP port
        acceptor=False,          # This endpoint initiates the connection
        multithreading=True,     # Allows concurrent send/receive operations
    )

    # Start the endpoint asynchronously (non-blocking)
    endpoint.start(blocking=False)
    print("start")

    # Send the encoded attack message to the remote receiver
    endpoint.send(attack_massage)
    print("sendet")

    # Gracefully shut down the connection and wait shortly
    endpoint.stop()
    sleep(1)



def pars_time(time_str):
    """
    Parses an ISO8601 datetime string and converts it into a timezone-aware datetime object.

    Raises:
        argparse.ArgumentTypeError: If the string is not a valid ISO8601 timestamp.
    """

    try:
       dt_naive = datetime.fromisoformat(time_str).astimezone() # without Timezone
       return dt_naive
    except ValueError:
        raise argparse.ArgumentTypeError(" invalid timeformat. Expected: YYYY-MM-DDTHH:MM:SS" )

def pars_ip(ip_str):
    """
    Validates an IP address string.
    
    Returns:
        IPv4Address or IPv6Address object.

    Raises:
        argparse.ArgumentTypeError: If the IP address is invalid.
    """
    try:
        return ipaddress.ip_address(ip_str)
    except ValueError:
        raise argparse.ArgumentTypeError("invalid ip-address")
    
def check_start_time(attack_start):
    """
    Ensures that a scheduled attack start time lies in the future.
    Prints a warning if the start time has already passed.
    """

    time_check=attack_start-datetime.now().astimezone()
    if time_check<= timedelta(0):
        print("Time is over! try a later starting.time")
  

if __name__ == "__main__":
     # Configure logging format and verbosity

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # Command-line interface definition
    pars = argparse.ArgumentParser(description = "Attack description")

    pars.add_argument("attack_plan", type = str, help = "JSON-File for coordination the data collection") 

    args = pars.parse_args()
    
     # Load attack definition from JSON file
    with open(args.attack_plan, "r", encoding="utf-8") as f:
        routemap= json.load(f)
    timer_check = 0 # Sum of attack + delay durations
    end = 0 # Total collection duration (only one per scenario)
    ip_list= [] # All IPs involved in the attack scenario

    # ------------------- Validate and analyze the attack plan -------------------
    for step in routemap:
        
        # Collection steps: executed only on one machine
        if step["type"] == "collection":
             time_stampe=pars_time(step["start"])
             check_start_time(time_stampe)
             end = end + step["duration"]
             pars_ip(step["ip"])
             ip_list.append(step["ip"])

        # Delay steps simply accumulate planned time     
        if step["type"] == "delay":
            timer_check = timer_check + step["duration"]
        if step["type"] == "attack":
            timer_check = timer_check + step["duration"]
            pars_ip(step["target"])
            pars_ip(step["source"])
            ip_list.append(step["target"])
            ip_list.append(step["source"])
    

    # Verify that total attack duration fits into the collection timeframe
    if end < timer_check:
        print("Schedule mismatch")
        exit

    # Remove duplicate IP addresses    
    ip_list =  list(set(ip_list))

    # Encode the routemap as JSON in bytes (for network transmission)
    body = json.dumps(routemap).encode()

     # ------------------- Trigger attack plan on all involved machines -------------------
    for ip in ip_list:
        initiat_attack_massages(ip, body)