"""

Author: Sandra Schneider
Modified: 15.05.2025
"""

import logging
from time import sleep
import socket

from datetime import datetime, timezone
from daisy.communication import connect_to_target, arpspoof_run, slowloris_run, reverse_shell_target, WebServer

from daisy.data_sources import DataHandler, PysharkProcessor, LivePysharkDataSource, \
    CSVFileRelay, EventHandler
import logging

import json



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

    attack_info = json.loads(msg.decode("utf-8"))
    start_attacke(attack_info)

def start_attacke(attack_info):

    # Lokale IP-Adresse
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)

   

    print(datetime.now(timezone.utc))
    print(attack_info)
    print(ip)
    time_check= 0
    duration_to_end=0

    t = WebServer()
    for step in attack_info:
        print(step)
        if step["type"] == "collection":
            time_to_wait= datetime.fromisoformat(step["start"]).astimezone()-datetime.now().astimezone()
            print(time_to_wait.total_seconds())
            sleep(time_to_wait.total_seconds())
            t.start()
            duration_to_end=step["duration"]
            
            if ip == step["ip"]:
                 relay = start_collection(step["file_name"])


             
        if step["type"] == "attack":
            time_check= time_check + step["duration"]
            if ip ==  step["target"]: 
                if step["name"]== "reverse_shell":
                    reverse_shell_target()    


            if ip == step["source"]:
                if step["name"] == "reverse_shell":
                    
                    connect_to_target(step["target"], step["source"])
        
                if step["name"] =="path_traversal":
                    print(step["target"])
                    print(step["duration"])
                    t.path_traversal(step["target"],step["duration"])
        
                if step["name"]=="arp_spoofing":
                    arpspoof_run(step["target"],step["source"],step["duration"])

                if step["name"]=="slowloris":
                   slowloris_run(step["target"],step["duration"])

            
           
        if step["type"]== "delay":
            time_check= time_check + step["duration"]
            print(f"sleep: {step["duration"]}")
            sleep(step["duration"])
        

    cool_down= duration_to_end-time_check
    print(cool_down)
    sleep(cool_down)
    relay.stop()
    t.shutdown()
    print("thread stoppen")
    t.join()
    print("thread gestoppt")



    sleep(30)

   
def start_collection(file_name): #name, start time, end time, lable, target, source

    source = LivePysharkDataSource()
    #events = EventHandler().append_event(label= attack_info[3], condition = )
    processor = PysharkProcessor().packet_to_dict()
    handler = DataHandler(data_source=source, data_processor=processor, multithreading=True)
    relay = CSVFileRelay(target_file= file_name+".csv", data_handler=handler, overwrite_file=True, separator=";")

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    relay.start(blocking=False)

    return relay
    
    



def message_to_list(msg):
    #name, start time, end time, lable, target, source
    attack_info = msg.split("§",5)
    return attack_info

if __name__ == "__main__":
    
    single_message_acceptor()