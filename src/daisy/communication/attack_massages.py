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
    :param taget_ip: ip address of the target e.g. "127.0.0.1"
    :param attack_massage: Generated message with attack definition
    """
    
    endpoint = StreamEndpoint(
        name="attack_massage",
        remote_addr=(ip, 32000), 
        acceptor=False,
        multithreading=True,
    )
    endpoint.start(blocking=False)
    print("start")

    endpoint.send(attack_massage)
    print("sendet")


    endpoint.stop()
    sleep(1)



def pars_time(time_str):
    try:
       dt_naive = datetime.fromisoformat(time_str).astimezone() # without Timezone
       return dt_naive
    except ValueError:
        raise argparse.ArgumentTypeError(" invalid timeformat. Expected: YYYY-MM-DDTHH:MM:SS" )

def pars_ip(ip_str):
    try:
        return ipaddress.ip_address(ip_str)
    except ValueError:
        raise argparse.ArgumentTypeError("invalid ip-address")
    
def check_start_time(attack_start):
    time_check=attack_start-datetime.now().astimezone()
    if time_check<= timedelta(0):
        print("Time is over! try a later starting.time")
  

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    pars = argparse.ArgumentParser(description = "Attack description")

    pars.add_argument("attack_plan", type = str, help = "JSON-File for coordination the data collection") 

    args = pars.parse_args()
    

    with open(args.attack_plan, "r", encoding="utf-8") as f:
        routemap= json.load(f)
    timer_check = 0
    end = 0 # limitierung das nur auf einer seite aufgezeichnet wird... da end sonnst überschrieben wird... idee?
    ip_list= []
    for step in routemap:
        
        if step["type"] == "collection":
             time_stampe=pars_time(step["start"])
             check_start_time(time_stampe)
             end = end + step["duration"]
             pars_ip(step["ip"])
             ip_list.append(step["ip"])
        if step["type"] == "delay":
            timer_check = timer_check + step["duration"]
        if step["type"] == "attack":
            timer_check = timer_check + step["duration"]
            pars_ip(step["target"])
            pars_ip(step["source"])
            ip_list.append(step["target"])
            ip_list.append(step["source"])
    
    if end < timer_check:
        print("Zeitplan haut nicht hin")
        exit
        
    ip_list =  list(set(ip_list))

    body = json.dumps(routemap).encode()

    for ip in ip_list:
        initiat_attack_massages(ip, body)