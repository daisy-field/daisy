"""

Author: Sandra Schneider
Modified: 15.05.2025
"""

import logging
from time import sleep
import socket
from datetime import datetime, timezone
from daisy.communication import connect_to_target, arpspoof_run, start_webserver, path_traversal, slowloris_run, reverse_shell_target

from daisy.data_sources import DataHandler, PysharkProcessor, LivePysharkDataSource, \
    CSVFileRelay, EventHandler
import logging



from daisy.communication import StreamEndpoint

#name, start time, end time, lable, target, source

def single_message_acceptor():
    """Creates and starts an acceptor to perform a single receive before stopping the
    endpoint, to test if endpoints can be stopped while they are receiving multiple
    messages.
    """
    endpoint = StreamEndpoint(
        name="Acceptor",
        addr=("0.0.0.0", 32000), 
        acceptor=True,
        multithreading=False,
        
    )
    endpoint.start()
      
    try:
        msg = endpoint.receive(5)
    except TimeoutError:
        print("nothing to receive")
        exit(-1)
    sleep(2)

    endpoint.stop()

    attack_info = message_to_list(msg)#name, start time, end time, lable, target, source

    # Lokale IP-Adresse
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)

    time_to_wait= datetime.fromisoformat(attack_info[1])-datetime.now(timezone.utc)
    print(datetime.now(timezone.utc))
    print(time_to_wait)

    print(attack_info)
    print(ip)
       


    if attack_info[4] == ip or attack_info[4]== "127.0.0.1":
        print( attack_info[0]+" at "+ attack_info[1]+" from "+ attack_info[4])
        relay_target = start_collection(attack_info)
        sleep(time_to_wait.total_seconds())
        if attack_info[0]== "reverse_shell":
            reverse_shell_target()
        
        if attack_info[0] == ("path_traversal" or "slowloris"):
            start_webserver()

        time_to_stop= datetime.fromisoformat(attack_info[2])-datetime.now(timezone.utc)
        sleep(time_to_stop.total_seconds())
        relay_target.stop()

        
        

    elif attack_info[5] == ip or attack_info[5]=="127.0.0.1": 
        print(attack_info[0]+" at "+ attack_info[1]+" to"+ attack_info[5])
        relay_source = start_collection(attack_info)
        sleep(time_to_wait.total_seconds())
        if attack_info[0] == "reverse_shell":
            sleep(30)
            connect_to_target(attack_info[4], attack_info[5])
        
        if attack_info[0] =="path_traversal":
            sleep(30)
            path_traversal(attack_info[4])
        
        if attack_info[0]=="arp_spoofing":
            arpspoof_run(attack_info[4],attack_info[5])

        if attack_info[0]=="slowloris":
            sleep(30)
            slowloris_run(attack_info[4])

        time_to_stop= datetime.fromisoformat(attack_info[2])-datetime.now(timezone.utc)
        sleep(time_to_stop.total_seconds())
        relay_source.stop()

    
    else:
        print("Im not target or source!")
        exit(-1)

    sleep(30)
   
def start_collection(attack_info): #name, start time, end time, lable, target, source

    source = LivePysharkDataSource()
    #events = EventHandler().append_event(label= attack_info[3], condition = )
    processor = PysharkProcessor().packet_to_dict()
    handler = DataHandler(data_source=source, data_processor=processor)
    relay = CSVFileRelay(target_file= attack_info[0]+".csv", data_handler=handler, overwrite_file=True, separator=";")

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )

    relay.start(blocking=False)

    return relay
    
    



def message_to_list(msg):
    #name, start time, end time, lable, target, source
    attack_info = msg.split("§",5)
    return attack_info

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )
    single_message_acceptor()