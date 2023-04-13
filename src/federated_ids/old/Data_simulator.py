#
#   Created: 2023
#   Author: Seraphin Zunzer
#   License: MIT License
# ---------------------------------------------------------------


# Optimized data simulator for real-world dataset.
# Data simulator, which reads in the real-world dataset, applies preprocessing and sends
# the samples to the client one after the other.


import ipaddress
import json
import math
import os
import socket
import sys
import pandas as pd
import time

HOST = "127.0.0.1"  # The server's hostname or IP address
CLIENT_PORT = None  # The port used by the client, set by cli
import re

length_limit = 70

def load_dataset(path):
    client_df = pd.read_csv(path, lineterminator='\n',usecols =['meta.len', 'meta.time', 'meta.protocols', 'ip.addr', 'sll.halen', 'sll.pkttype', 'sll.eth', 'sll.hatype', 'sll.unused', 'ipv6.addr', 'ipv6.plen', 'ipv6.tclass', 'ipv6.flow', 'ipv6.dst', 'ipv6.nxt', 'ipv6.src_host', 'ipv6.host', 'ipv6.hlim', 'tcp.window_size_scalefactor', 'tcp.checksum.status', 'tcp.analysis.bytes_in_flight', 'tcp.analysis.push_bytes_sent', 'tcp.payload', 'tcp.port', 'tcp.len', 'tcp.hdr_len', 'tcp.window_size', 'tcp.checksum', 'tcp.ack', 'tcp.srcport', 'tcp.stream', 'tcp.dstport', 'tcp.seq', 'tcp.window_size_value', 'tcp.status', 'tcp.urgent_pointer', 'tcp.nxtseq', 'data.data', 'data.len', 'tcp.analysis.acks_frame', 'tcp.analysis.ack_rtt', 'sll.ltype', 'cohda.Type', 'cohda.Ret', 'cohda.llc.MKxIFMsg.Ret', 'ipv6.addr', 'ipv6.dst', 'ipv6.plen', 'tcp.stream', 'tcp.payload', 'tcp.urgent_pointer', 'tcp.port', 'tcp.options.nop', 'tcp.options.timestamp', 'tcp.flags', 'tcp.window_size_scalefactor', 'tcp.dstport', 'tcp.len', 'tcp.checksum', 'tcp.window_size', 'tcp.srcport', 'tcp.checksum.status', 'tcp.nxtseq', 'tcp.status', 'tcp.analysis.bytes_in_flight', 'tcp.analysis.push_bytes_sent', 'tcp.ack', 'tcp.hdr_len', 'tcp.seq', 'tcp.window_size_value', 'data.data', 'data.len', 'tcp.analysis.acks_frame', 'tcp.analysis.ack_rtt', 'eth.src.addr', 'eth.src.eth.src_resolved', 'eth.src.ig', 'eth.src.src_resolved', 'eth.src.addr_resolved', 'ip.proto', 'ip.dst_host', 'ip.flags', 'ip.len', 'ip.checksum', 'ip.checksum.status', 'ip.version', 'ip.host', 'ip.status', 'ip.id', 'ip.hdr_len', 'ip.ttl'])
    client_df['meta.time'] = pd.to_datetime(client_df['meta.time']) # read selected columns from dataset
    # convert time to timestamp

    client_df.insert(3, "year", client_df['meta.time'].dt.year.astype(int) )    # separate timestamps
    client_df.insert(4, "month",client_df['meta.time'].dt.month.astype(int) )
    client_df.insert(5, "day", client_df['meta.time'].dt.day.astype(int))
    client_df.insert(6, "hour", client_df['meta.time'].dt.hour.astype(int) )
    client_df.insert(7, "minutes",client_df['meta.time'].dt.minute.astype(int) )
    client_df.insert(8, "seconds",client_df['meta.time'].dt.second.astype(int) )

    del client_df['meta.time']

    client_data = client_df.to_numpy()  # convert data into array
    ip_idx = client_df.columns.get_loc("ip.addr")   # get the indices of ip and protocol since they are used to determine the anomalies
    prot_idx = client_df.columns.get_loc("meta.protocols")

    return client_data, ip_idx ,prot_idx

def natural_sort(l):    # function to sort the dataset directory to ensure the correct order
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def send_sample_to_client(packet, label):
    """Send new weights back to server"""
    bytes_weights_data = json.dumps([packet, [label]]).encode()    # put data int lists and convert them into json
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, CLIENT_PORT))
            s.sendall(bytes_weights_data) # send data to connected client
    except Exception as e:
        print(f"Data Source failed to connect to Client {CLIENT_ID}, {CLIENT_PORT}")


if __name__ == "__main__":
    CLIENT_ID = int(sys.argv[1])
    CLIENT_PORT = int(sys.argv[2])
    DATASET_PATH = f'datasets/cohda_box_{CLIENT_ID}'
    time.sleep(1)
    k=0

    for filename in natural_sort(os.listdir(DATASET_PATH)):
        print("Reading " + filename + "...")
        name, file_extension = os.path.splitext(filename)
        if 'csv' in file_extension:

            packets, ip_idx, prot_idx = load_dataset(f'datasets/cohda_box_{CLIENT_ID}/{filename}')

            if len(packets[0]) < length_limit: #skip if packet is malformed
                print(len(packets[0]))
                continue

            for i in range(len(packets)):
                sample = packets[i]
                conv_sample = list()

                for item in sample:
                    if isinstance(item, str):   # if item is string
                        try:
                            res = int(ipaddress.IPv4Address(item))  # try to replace ip
                            conv_sample.append(res)
                        except:
                            try:
                                res = int(item, 16) # try to convert from string to hex
                                conv_sample.append(res)
                            except:
                                res = hash(item)
                                conv_sample.append(int(hash(res)))  # if failed, calculate hash value

                    elif not math.isfinite(item):   # if value is infinity replace with 0
                        conv_sample.append(0)

                    elif isinstance(item, int) or isinstance(item, float) : # otherwise append value
                        conv_sample.append(float(item))
                    else:
                        print("Failed to convert", item)
                        conv_sample.append(0)


                try:    # check if current sample is anomaly to send the correct label
                    if CLIENT_ID == 5 and sample[4] == 6 and sample[5] == 12 and (34<=sample[6]<=40) and  ("tcp" in sample[prot_idx] or "http" in sample[prot_idx]) and "192.168.213.86" in sample[ip_idx] and "185." in sample[ip_idx]:
                        send_sample_to_client(conv_sample[:length_limit], "Installation Attack Tool")
                        print("- Installation of Attack tool")
                    elif CLIENT_ID == 5 and sample[4] == 6 and (12 <= sample[5] <= 13) and (49<=sample[6] or sample[6]<=23) and ("tcp" in sample[prot_idx] or "ssh" in sample[prot_idx]) and "192.168.230.3" in sample[ip_idx] and "192.168.213.86" in sample[ip_idx]:
                        send_sample_to_client(conv_sample[:length_limit], "SSH Brute Force")
                        print("- SSH Brute Force")
                    elif CLIENT_ID == 5 and  sample[4] == 6 and (sample[5] == 13) and (25<=sample[6]<=32) and ("tcp" in sample[prot_idx] or "ssh" in sample[prot_idx]) and "192.168.230.3" in sample[ip_idx] and "192.168.213.86" in sample[ip_idx]:
                        send_sample_to_client(conv_sample[:length_limit], "SSH Privilege Escalation")
                        print("- SSH Privilege Escalation")
                    elif CLIENT_ID == 2 and  sample[4] == 6 and (12 <= sample[5] <= 13) and (49 <= sample[6] or sample[6] <= 23) and ("tcp" in sample[prot_idx] or "ssh" in sample[prot_idx]) and "192.168.230.3" in sample[ip_idx] and "130.149.98.119" in sample[ip_idx]:
                        print("- SSH Brute Force Response")
                        send_sample_to_client(conv_sample[:length_limit], "SSH Brute Force Response")
                    elif CLIENT_ID == 2 and sample[4] == 6 and (sample[5] == 13) and (25 <= sample[6] or sample[6] <= 32) and ("tcp" in sample[prot_idx] or "ssh" in sample[prot_idx]) and "192.168.230.3" in sample[ip_idx] and "130.149.98.119" in sample[ip_idx]:
                        print("- SSH Data leakage")
                        send_sample_to_client(conv_sample[:length_limit], "SSH  Data leakage")
                    else:
                        if not len(conv_sample) == length_limit:
                            conv_sample = conv_sample[:length_limit]
                        send_sample_to_client(conv_sample, "BENIGN")
                        print("- Normal")
                except:
                    if not len(conv_sample) == length_limit:
                        conv_sample = conv_sample[:length_limit]
                    send_sample_to_client(conv_sample, "BENIGN")
                    print("Normal")

                if i % 20000 == 0:
                    print("PROCESSED 20000 DATA:", i," \n\n\n\n")
            print("FInished ", filename)
    print("FINISHED SIMULATOR\n")

