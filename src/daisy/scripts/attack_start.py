"""

Author: Sandra Schneider
Modified: 20.11.2025
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
    """
    Creates and starts a StreamEndpoint that receives exactly one message.
    After receiving the message (or timing out), the endpoint is stopped.
    This function is used to test whether an endpoint can be safely stopped
    while it is still capable of receiving multiple messages.
    """
     # Create a TCP-based endpoint that listens for incoming connections
    endpoint = StreamEndpoint(
        name="Acceptor",
        addr=("0.0.0.0", 32000), 
        acceptor=True,    # Accepts incoming TCP connections    
        multithreading=False,   # Single-threaded communication
        
    )
    endpoint.start()   # Start listening for incoming connections
      
    try:
        msg = endpoint.receive(5)
    except TimeoutError:
        print("nothing to receive")
        exit(-1)


    sleep(2)     # Small delay before shutting down the endpoint
    endpoint.stop()      # Gracefully stop the network endpoint

    attack_info = json.loads(msg.decode("utf-8"))   # Convert the received bytes to JSON
    start_attacke(attack_info)       # Begin execution of the attack scenario

def start_attacke(attack_info):
    """
    Executes the attack sequence provided in attack_info.
    The structure contains steps of types: 'collection', 'attack', and 'delay'.
    Each step defines what should happen at what time and on which machine.
    """

    # Determine the local machine's IP address
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)

   

    print(datetime.now(timezone.utc))
    print(attack_info)
    print(ip)


    time_check= 0   # Tracks total duration of executed attack steps
    duration_to_end=0    # Duration of the collection phase

    t = WebServer()     # Web server instance used for certain attacks

    # Iterate over all attack steps (schedule)
    for step in attack_info:
        print(step)


         # --------------------------- COLLECTION STEP ---------------------------
        if step["type"] == "collection":

            # Calculate waiting time until the scheduled start timestamp
            time_to_wait= datetime.fromisoformat(step["start"]).astimezone()-datetime.now().astimezone()
            print(time_to_wait.total_seconds())
            sleep(time_to_wait.total_seconds())

            # Start the associated web server
            t.start()
            duration_to_end=step["duration"]

             # If the collection is meant for this system → start packet capture
            if ip == step["ip"]:
                 relay = start_collection(step["file_name"])

        # --------------------------- ATTACK STEP -------------------------------      
        if step["type"] == "attack":
            time_check= time_check + step["duration"]

            # Actions executed on the victim system
            print(step["target"])
            if ip ==  step["target"]: 
                print(step["name"])
                if step["name"]== "reverse_shell":
                    
                    reverse_shell_target() 

                sleep(step["duration"])

            # Actions executed on the attacking system
            if ip == step["source"]:
                if step["name"] == "reverse_shell":
                    
                    connect_to_target(step["target"], step["source"])
        
                if step["name"] =="path_traversal":
                    print(step["target"])
                    print(step["duration"])
                    t.path_traversal(step["target"],step["duration"])
        
                if step["name"]=="arp_spoofing":
                    arpspoof_run(step["target"],"192.168.178.1",step["duration"])

                if step["name"]=="slowloris":
                   slowloris_run(step["target"],step["duration"])

            
        # --------------------------- DELAY STEP --------------------------------
        if step["type"]== "delay":
            time_check= time_check + step["duration"]
            print(f"sleep: {step["duration"]}")
            sleep(step["duration"])
        
    # Compute remaining time of the collection phase after attacks finished
    cool_down= duration_to_end-time_check
    print(cool_down)
    sleep(cool_down)

     # Stop packet capture
    relay.stop()

     # Shut down the web server and wait for its worker thread
    t.shutdown()
    print("thread stoppen")
    t.join()
    print("thread gestoppt")

    sleep(30) # Additional waiting period before ending the process

   
def start_collection(file_name): #name, start time, end time, lable, target, source

    """
    Starts a live packet capture using pyshark and writes the results into a CSV file.
    Returns the relay object, which can later be stopped.
    """
    source = LivePysharkDataSource()
    #events = EventHandler().append_event(label= attack_info[3], condition = )  #for live labeling
    processor = PysharkProcessor().packet_to_dict()

    handler = DataHandler(data_source=source, data_processor=processor, multithreading=True)

    # CSV relay writes captured packets into a file
    relay = CSVFileRelay(target_file= file_name+".csv", data_handler=handler, overwrite_file=True, separator=";")

    # Configure logging output
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    # Non-blocking start → continues execution immediately
    relay.start(blocking=False)

    return relay
    
    

def message_to_list(msg):
    """
    Splits a custom message format using '§' as delimiter.
    Returns a list containing the extracted fields.
    """

    #name, start time, end time, lable, target, source
    attack_info = msg.split("§",5)
    return attack_info

if __name__ == "__main__":
    # Entry point: wait for and receive exactly one message
    single_message_acceptor()