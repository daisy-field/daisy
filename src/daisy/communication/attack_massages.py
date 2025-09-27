"""send messages to coordinate attacks and their recording.

Author: Sandra Schneider
Modified: 14.05.2025
"""

import logging

import argparse
"""send messages to coordinate attacks and their recording.

Author: Sandra Schneider
Modified: 14.05.2025
"""

import logging

import argparse
import ipaddress
from zoneinfo import ZoneInfo
from time import sleep
from datetime import datetime, timezone, timedelta

from daisy.communication import StreamEndpoint


def initiat_attack_massages(ip    , attack_massage,x):
    """ 
    :param taget_ip: ip address of the target e.g. "127.0.0.1"
    :param attack_massage: Generated message with attack definition
    """
    
    endpoint = StreamEndpoint(
        name="attack_massage",
        remote_addr=(ip, 32000+x), #anderer port?
        acceptor=False,
        multithreading=True,
    )
    endpoint.start(blocking=False)
    print("start")

    endpoint.send(attack_massage)
    print("sendet")


    endpoint.stop()
    sleep(1)


def generate_massage(attack_name, attack_start, attack_end, attack_type, target, source):
    """ 
    :param attack_name: Name of attack
    :param attack_start: Time when the attack starts
    :param attack_end: Time when the attack ends
    :param attack_type: Type of attack and MITRE ATT&CK ID
    :param target: Target of attack
    :param source: Source of attack
    """
    return f"{attack_name}§{attack_start}§{attack_end}§{attack_type}§{target}§{source}"

def pars_time(time_str, timezone="Europe/Berlin"):
    try:
       dt_naive = datetime.fromisoformat(time_str) # without Timezone
       return dt_naive#.replace(tzinfo=ZoneInfo(timezone)) 
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
    pars.add_argument("attack_start", type = str, help = "Time when the attack starts ISO8601")
    pars.add_argument("attack_end", type = str, help = "Time when the attack ends ISO8601") 
    pars.add_argument("attack_type", type = str, help = "Type of attack and MITRE ATT&CK ID")
    pars.add_argument("target", type = str, help = "Target of attack ipv4 or ipv6") 
    pars.add_argument("source", type = str, help = "Source of attack ipv4 or ipv6")
    pars.add_argument("timezone", type = str, help = "timezone from target and source. default Europe/Berlin")

    args = pars.parse_args()

    if not args.timezone:
        attack_start= pars_time(args.attack_start)
        attack_end = pars_time(args.attack_end)
    else:
        attack_start= pars_time(args.attack_start, timezone=args.timezone)
        attack_end= pars_time(args.attack_end, timezone=args.timezone)

    time_check=attack_start-datetime.now(timezone.utc)
    if time_check<= timedelta(0):
        print("Time is over! try a later starting.time")
    if attack_end-attack_start<= timedelta(0):
        print("the End can't be bevor start!")


    target = pars_ip(args.target)
    source = pars_ip(args.source)


    msg = generate_massage(args.attack_name, attack_start, attack_end, args.attack_type, target, source)

    

    initiat_attack_massages(str(target), msg,0)

    initiat_attack_massages(str(source), msg,1)