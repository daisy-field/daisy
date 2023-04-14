"""
    TODO

    Author: Jonathan Ackerschewski, Fabian Hofmann
    Modified: 13.04.22
"""

import argparse
import datetime
import logging
import os
import re
from time import time

import pyshark
from pyshark import capture
from pyshark.capture.capture import TSharkCrashException

import src.communication.message_stream as stream

# TODO further cleanup, docstrings, typehints


def pcap2dict(pcap):
    try:
        pcap.load_packets()
    except TSharkCrashException:
        return None
    pcap.reset()
    packet_count = len(pcap)
    step_size = packet_count // 20
    cur_packet = 0

    logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.DEBUG)
    endpoint = stream.StreamEndpoint(addr=("127.0.0.1", 13000), remote_addr=("127.0.0.1", 12000),
                                     endpoint_type=stream.SOURCE, multithreading=False, buffer_size=10000)
    endpoint.start()

    while True:
        try:
            if cur_packet % step_size == 0:
                print(f"Reading packet {cur_packet} of {packet_count}...")
            cur_packet += 1
            packet = pcap.next_packet()
            endpoint.send(packet)
        except StopIteration:
            break

    endpoint.stop()


def pcap_convert(pcap_files):
    failed_files = []
    durations = []
    overall_start_time = time()
    for i in range(len(pcap_files)):
        start_time = time()
        file = pcap_files[i]
        print(f"Starting file {file}")
        try_counter = 0
        dict_list = None
        while dict_list is None and try_counter < 10:
            print("Trying to read file...")
            pcap = pyshark.FileCapture(file)
            dict_list = pcap2dict(pcap)
            try_counter += 1
            break
        if dict_list is None:
            print(f"Failed to read file '{file}'. Skipping...")
            failed_files += [file]
            continue

        end_time = time()
        durations += [end_time - start_time]
        files_left = len(pcap_files) - i - 1
        eta = (sum(durations[-10:]) / len(durations[-10:])) * files_left
        eta = datetime.timedelta(seconds=int(eta), microseconds=int(eta * 1000000) % 1000000)
        print(f"File {file} finished in {durations[-1]} seconds. {files_left} files left. ETA: {eta}")

    overall_end_time = time()
    duration = overall_end_time - overall_start_time
    duration = datetime.timedelta(seconds=int(duration), microseconds=int(duration * 1000000) % 1000000)
    print(f"Finished in {duration}. Failed files: {failed_files}")


if __name__ == '__main__':
    # Create arguments and descriptions
    p = argparse.ArgumentParser(description="Reads a pcap file or a directory containing pcap files and converts them"
                                            " to a csv file")
    p.add_argument("pcap_path", metavar="pcap", type=str, nargs='+', help="The pcap files or the paths to directories"
                                                                          " containing pcap files. If directory is"
                                                                          " given, only files ending in .pcap are"
                                                                          " considered")
    p.add_argument("-o", "--output", metavar="csv", dest="output_file", type=str, help="The path to the output file, "
                                                                                       "where all input files will be "
                                                                                       "merged into. If not specified "
                                                                                       "an output file for each input "
                                                                                       "will be created with the same "
                                                                                       "name as the input file")
    p.add_argument("-j", "--json", dest="as_json", action="store_true", help="Stores the result in json form instead"
                                                                             " of csv")
    p.add_argument("-f", "--force", dest="force", action="store_true", help="Forces override of existing files")

    args = p.parse_args()

    # Search the given paths for all .pcap files, if the path was a directory or check if the file exists if the path
    # was a file.
    pcap_files = []
    for path in args.pcap_path:
        if os.path.isdir(path):
            dirs = [(x[0], x[2]) for x in os.walk(path)]
            files = [os.path.join(y[0], x) for y in dirs for x in y[1] if x.endswith(".pcap")]
            if not files:
                print(f"Directory '{path}' does not contain any .pcap files.")
                exit(1)
            pcap_files += files
        elif os.path.isfile(path):
            pcap_files.append(path)
        else:
            if args.force:
                os.remove(path)
            else:
                print(f"File '{path}' does not exist.")
                exit(1)

    if not pcap_files:
        print(f"No files could be found.")
        exit(1)

    # Check if the output file already exists, if it is given or if any of the output files exist, if it was not
    # specified.
    output_file = None
    if args.output_file is not None:
        output_file = args.output_file
        if not args.as_json and not output_file.endswith(".csv"):
            output_file += ".csv"
        elif args.as_json and not output_file.endswith(".json"):
            output_file += ".json"
        if os.path.exists(output_file):
            if args.force:
                os.remove(output_file)
            else:
                print(f"Output file '{output_file}' already exists.")
                exit(1)
    else:
        for file in pcap_files:
            cur_file = re.sub("\.pcap$", '', file)
            if args.as_json:
                cur_file += ".json"
            else:
                cur_file += ".csv"
            if os.path.exists(cur_file):
                if args.force:
                    os.remove(cur_file)
                else:
                    print(f"Output file '{cur_file}' already exists.")
                    exit(1)

    # Start conversion with the given files and output file, if specified
    pcap_convert(pcap_files)
