import argparse
import datetime
import json
import os
import re
from collections.abc import MutableMapping
from time import time

import pandas as pd
import pyshark


def pcap2dict(pcap):
    packet_json = []
    try:
        pcap.load_packets()
    except pyshark.capture.capture.TSharkCrashException:
        return None
    pcap.reset()
    packet_count = len(pcap)
    step_size = packet_count // 20
    cur_packet = 0
    while True:
        try:
            if cur_packet % step_size == 0:
                print(f"Reading packet {cur_packet} of {packet_count}...")
            cur_packet += 1
            packet = pcap.next_packet()
            packet_dict = {}

            meta_dict = {
                "number": packet.number,
                "len": packet.length,
                "protocols": [x.layer_name for x in packet.layers],
                "time_epoch": packet.sniff_timestamp,
                "time": str(packet.sniff_time)
            }
            packet_dict.update({"meta": meta_dict})

            for layer in packet.layers:
                packet_dict.update(addLayerToDict(layer))

            packet_json += [packet_dict]
        except StopIteration:
            break

    return packet_json


def addLayerToDict(layer):
    if isinstance(layer, pyshark.packet.layers.json_layer.JsonLayer):
        dict = {}
        for field in layer.field_names:
            result = addLayerToDict(layer.get_field(field))
            if isinstance(result, list):
                if hasattr(layer.get_field(field), "layer_name"):
                    dict.update({layer.get_field(field).layer_name: result})
                else:
                    # dict.update({f"RandomFieldName{randint(100000, 999999)}": result})
                    # Assumptions:
                    # - result has no layer_name, because it is a list
                    # - the list is never empty
                    # - all elements inside the list are dictionaries
                    # - all dictionaries inside the list have one element
                    # - all dictionaries inside the list share the same key
                    # If this fails, try enabling the RandomFieldName above and look in the generated json
                    # what the result looks like and which assumption doesn't hold
                    dict.update({next(iter(result[0].keys())): [res[next(iter(result[0].keys()))] for res in result]})
            else:
                dict.update(result)
        layer_list = {layer.layer_name: dict}
        return layer_list
    elif isinstance(layer, pyshark.packet.fields.LayerFieldsContainer):
        dict = {}
        if len(layer.all_fields) > 1:
            print(f"Layer count: {len(layer.all_fields)}, layer: {layer.all_fields}")
        for field in layer.all_fields:
            dict.update(addLayerToDict(field))
        return dict
    elif isinstance(layer, pyshark.packet.fields.LayerField):
        return {layer.name: layer.raw_value}
    elif isinstance(layer, list):
        l = []
        for sub_layer in layer:
            l += [addLayerToDict(sub_layer)]
        return l


def print2file(pcap_data, pcap_files, output_file, as_json):
    '''
    Writes the given json to to file or converts it to csv before writing if as_json is false
    :param json_list: the json as list of dictionaries
    :param pcap_files: the pcap_files paths or path of a single file if output_file is None and each json/csv should
    have its own output
    :param output_file: Path to the output file or None
    :param as_json: Write as json or csv? (True/False)
    :return:
    '''
    file_path = output_file
    if not output_file:
        file_path = re.sub("\.pcap$", '', pcap_files)
        if as_json:
            file_path += ".json"
        else:
            file_path += ".csv"

    if as_json:
        text = str(pcap_data)
        text = text.replace("'", "")
        text = text.replace("\\n", "\n")

        with open(file_path, "w") as f:
            f.write(text)
    else:
        df = pd.DataFrame.from_dict(pcap_data)
        df.to_csv(file_path, index=False)


def flatten_dict(dict, par_key=""):
    seperator = "."
    items = {}
    for key, val in dict.items():
        cur_key = par_key + seperator + key if par_key else key
        if isinstance(val, MutableMapping):
            items.update(flatten_dict(val, cur_key))
        else:
            items.update({cur_key: val})
    return items


def dict2flat_dict(dict):
    dict_list = []
    for d in dict:
        dict_list += [flatten_dict(d)]
    return dict_list


def dict2json(dict):
    return json.dumps(dict, indent=2)


def pcap_convert(pcap_files, output_file, as_json):
    failed_files = []
    durations = []
    overall_start_time = time()
    packet_list = []
    for i in range(len(pcap_files)):
        start_time = time()
        file = pcap_files[i]
        print(f"Starting file {file}")
        try_counter = 0
        dict_list = None
        while dict_list is None and try_counter < 10:
            print("Trying to read file...")
            pcap = pyshark.FileCapture(file, use_json=True)  # , include_raw=True)
            dict_list = pcap2dict(pcap)
            try_counter += 1
        if dict_list is None:
            print(f"Failed to read file '{file}'. Skipping...")
            failed_files += [file]
            continue

        if as_json:
            pcap_data = dict2json(dict_list)
        else:
            pcap_data = dict2flat_dict(dict_list)

        if not output_file:
            print2file(pcap_data, file, output_file, as_json)
        else:
            packet_list += pcap_data
        end_time = time()
        durations += [end_time - start_time]
        files_left = len(pcap_files) - i - 1
        eta = (sum(durations[-10:]) / len(durations[-10:])) * files_left
        eta = datetime.timedelta(seconds=int(eta), microseconds=int(eta * 1000000) % 1000000)
        print(f"File {file} finished in {durations[-1]} seconds. {files_left} files left. ETA: {eta}")

    if output_file:
        print2file(packet_list, pcap_files, output_file, as_json)

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
    pcap_convert(pcap_files, output_file, args.as_json)
