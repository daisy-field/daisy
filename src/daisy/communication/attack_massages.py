"""send messages to coordinate attacks and their recording.

Author: Sandra Schneider
Modified: 14.05.2025
"""

import logging
import datetime
import argparse
import ipaddress
from zoneinfo import ZoneInfo
from time import sleep

from daisy.communication import StreamEndpoint


def initiat_attack_massages_taget(target_ip, attack_massage):
    """ 
    :param taget_ip: ip address of the target e.g. "127.0.0.1"
    :param attack_massage: Generated message with attack definition
    """
    endpoint = StreamEndpoint(
        name="target_massage",
        addr=(target_ip, 13000), # anderer port?
        remote_addr=(target_ip, 32000), #anderer port?
        acceptor=False,
        multithreading=True,
        buffer_size=10000,
    )
    endpoint.start()

    endpoint.send(attack_massage)
    try:
        endpoint.receive(5)
    except TimeoutError:
        print("Can't inform target!")
    sleep(2)

    endpoint.stop()

def generate_massage(attack_name, attack_start, attack_end, attack_type, target, source):
    """ 
    :param attack_name: Name of attack
    :param attack_start: Time when the attack starts
    :param attack_end: Time when the attack ends
    :param attack_type: Type of attack and MITRE ATT&CK ID
    :param target: Target of attack
    :param source: Source of attack
    """
    return f"{attack_name}§{attack_start}§{attack_end}§{attack_type}§{target}${source}"

def pars_time(time_str, timezone="Europe/Berlin"):
    try:
        dt_naive = datetime.datetime.fromisoformat(time_str) # without Timezone
        return dt_naive.replace(tzinfo=ZoneInfo(timezone)) 
    except ValueError:
        raise argparse.ArgumentTypeError(" invalid timeformat. Expected: YYYY-MM-DDTHH-MM-SS" )

def pars_ip(ip_str):
    try:
        return ipaddress.ip_address(ip_str)
    except ValueError:
        raise argparse.ArgumentTypeError("invalid ip-address")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    pars = argparse.ArgumentParser(description = "Attack description")

    pars.add_argument("attack_name", type = str, help = "Name of attack") 
    pars.add_argument("attack_start", type = parse_time, help = "Time when the attack ends ISO8601")
    pars.add_argument("attack_end", type = parse_time, help = "Time when the attack ends ISO8601") 
    pars.add_argument("attack_type", type = str, help = "Type of attack and MITRE ATT&CK ID")
    pars.add_argument("target", type = parse_ip, help = "Target of attack ipv4 or ipv6") 
    pars.add_argument("source", type = parse_ip, help = "Source of attack ipv4 or ipv6")
    pars.add_argument("timezone", type = str, help = "timezone from target and source. default Europe/Berlin")

    args = pars.parse_args()

    if(args.timezone==None):
        attack_start= pars_time(args.attack_start)
        attack_end = pars_time(args.attack_end)
    else:
        attack_start= pars_time(args.attack_start, timezone=args.timezone)
        attack_end= pars_time(args.attack_end, timezone=args.timezone)

    target = pars_ip(args.target)
    pars_ip(args.source)


    msg = generate_massage(args.attack_name, attack_start, attack_end, args.attack_type, target, args.source)

    initiat_attack_massages_taget(target, msg)
