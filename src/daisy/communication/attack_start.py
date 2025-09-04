"""

Author: Sandra Schneider
Modified: 15.05.2025
"""

import logging
from time import sleep
import socket
from datetime import datetime, timezone
from .connect_to_target import *
from .arp_spoof import *
from .app import *
from .slowloris import *
from .reverse_shell_target import *
from ...daisy.scripts.temp import *


from daisy.communication import StreamEndpoint

#name, start time, end time, lable, target, source

def single_message_acceptor():
    """Creates and starts an acceptor to perform a single receive before stopping the
    endpoint, to test if endpoints can be stopped while they are receiving multiple
    messages.
    """
    endpoint = StreamEndpoint(
        name="Acceptor",
        addr=("127.0.0.1", 32000), #optional
        remote_addr=("127.0.0.1", 13000),#optional
        acceptor=None,
        multithreading=True,
        buffer_size=10000,
    )
    endpoint.start()
      
    try:
        msg = endpoint.receive(5)
    except TimeoutError:
        print("nothing to receive")
    sleep(2)

    endpoint.stop()

    attack_info = message_to_list(msg)#name, start time, end time, lable, target, source

    # Lokale IP-Adresse
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)

    time_to_wait= attack_info[1]-datetime.now(timezone.utc)

    
       


    if attack_info[4] == ip:
        print( attack_info[0]+" at "+ attack_info[1]+" from "+ attack_info[4])
        relay_target = start_collection(attack_info)
        sleep(time_to_wait)
        if attack_info[0]== "reverse_shell":
            reverse_shell_target()
        
        if attack_info[0] == ("path_traversal" or "slowloris"):
            start_webserver()

        
        

    elif attack_info[5] == ip:
        print(attack_info[0]+" at "+ attack_info[1]+" to"+ attack_info[5])
        relay_source = start_collection(attack_info)
        sleep(time_to_wait)
        if attack_info[0] == "reverse_shell":
            sleep(30)
            connect_to_target(attack_info[4], attack_info[5])
        
        if attack_info[0] =="path_traversal":
            sleep(30)
            path_traversal(attack_info[4])
        
        if attack_info[0]=="arp_spoofing":
            arpspoof(attack_info[4],attack_info[5])

        if attack_info[0]=="slowloris":
            sleep(30)
            main()

    
    else:
        print("Im not target or source!")

    sleep(300)
    relay_target.stop()
    relay_source.stop()



def message_to_list(msg):
    #name, start time, end time, lable, target, source
    attack_info = msg.split("§",5)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )