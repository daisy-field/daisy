from daisy.data_sources import CSVFileDataSource, DataProcessor, CSVFileRelay, DataHandler
import datetime

statistics_count = (0, 0, 0, 0, 0, 0, 0, {})  # Nr of packets in dataset, source bytes and destination bytes?, other bytes, source packets and destination packets, other packets, protocols
statistics_labels = {}  # Dict of Labels with their nr of packets
statistics_attacks = []  # List of attacks, where each attack is a tuple containing label, nr of packets in attack, start timestamp, end timestamp, source packets, dest packets, other packets, protocols, dict of ports and their counts
statistics_sips = {}  # Dict of source IPs and the nr of packets
statistics_dips = {}  # Dict of dest IPS and the nr of packets
statistics_ips = ({}, {}, {})  # Dict of Own and Others IPs
statistics_ports = ({}, {}, {})  # dict of ports and their counts for source and dest

ip_addresses = ["127.0.0.1","192.168.178.1","192.168.178.62","192.168.178.67"]  # TODO hier die eigenen IPs rein
mac_addresses = ["00:00:00:00:00:01"]


def add_label(p):
    p["label"] = "benign"
    return p


def percent(num, den):
    if den == 0:
        return 0.00
    return round(float(num) / float(den) * 100, 2)


def remove_empty(d):
    try:
        d.pop("")
    except:
        pass


def stats(d_point):
    global statistics_count
    global statistics_labels
    global statistics_attacks
    global statistics_sips
    global statistics_dips
    global statistics_ips
    global statistics_ports
    global ip_addresses
    global mac_addresses

    # TODO Hier die entsprechenden Features angeben
    ipsrc = "ip.src"
    ipdst = "ip.dst"
    iparpsrc = "arp.src.proto_ipv4"
    iparpdst = "arp.dst.proto_ipv4"
    src_ips = [d_point[ipsrc], d_point[iparpsrc]]  # Features, wo die IPs als source vorkommen
    dst_ips = [d_point[ipdst], d_point[iparpdst]]  # Features, wo die IPs als destination vorkommen
    tcpsrcport = "tcp.srcport"
    tcpdstport = "tcp.dstport"
    udpsrcport = "udp.port"
    udpdstport = "udp.dstport"
    sllsrc = "sll.src.eth"
    slldst = "sll.eth"

    is_src = False
    is_dst = False
    for address in ip_addresses:
        if address in src_ips:
            is_src = True
        if address in dst_ips:
            is_dst = True

    ports = (d_point[tcpsrcport], d_point[tcpdstport])
    switch_by_icmp = False
    if not ports[0]:
        ports = (d_point[udpsrcport], d_point[udpdstport])
    if is_dst:
        ports = (ports[1], ports[0])

    ip_address = (d_point[ipsrc], d_point[ipdst])
    if not ip_address[0]:
        ip_address = (d_point[iparpsrc], d_point[iparpdst])

    # For packets where we don't have an IP, we have to discover the src and dst using MAC.
    if not is_src and not is_dst:
        mac_found = False
        for mac in mac_addresses:
            if mac == d_point[sllsrc]:
                is_src = True
                mac_found = True
                break
        if not mac_found and d_point[sllsrc]:
            for mac in mac_addresses:
                if mac == d_point[slldst]:
                    is_dst = True
                    mac_found = True
                    break

    cur_counts = statistics_count
    cur_counts[7][d_point["meta.protocols"]] = cur_counts[7].get(
        d_point["meta.protocols"], 0) + 1
    statistics_count = (cur_counts[0] + 1,  # Packet count
                       cur_counts[1] + int(d_point["meta.len"]) if is_src else
                       cur_counts[1],  # source bytes
                       cur_counts[2] + int(d_point["meta.len"]) if is_dst else
                       cur_counts[2],  # dest bytes
                       cur_counts[3] + int(
                           d_point["meta.len"]) if not is_src and not is_dst else
                       cur_counts[3],  # other bytes
                       cur_counts[4] + 1 if is_src else cur_counts[4],
                       # source packet count
                       cur_counts[5] + 1 if is_dst else cur_counts[5],
                       # dest packet count
                       cur_counts[6] + 1 if not is_src and not is_dst else
                       cur_counts[6],  # other packets
                       cur_counts[7])  # Other protocol count

    cur_label = d_point["label"]
    statistics_labels[cur_label] = statistics_labels.get(cur_label,
                                                               0) + 1  # count of label

    if cur_label != "benign":
        new_attack = True
        for k in range(len(statistics_attacks)):
            attack = statistics_attacks[k]
            if attack[0] == cur_label and \
                    abs(attack[3] - float(d_point["meta.time_epoch"])) < 15:
                attack[7][d_point["meta.protocols"]] = attack[7].get(
                    d_point["meta.protocols"], 0) + 1
                if is_src or is_dst:
                    attack[8][ports[0]] = attack[8].get(ports[0], 0) + 1
                    attack[9][ports[1]] = attack[9].get(ports[1], 0) + 1
                if not is_src and not is_dst:
                    attack[10][ports[0]] = attack[10].get(ports[0], 0) + 1
                    attack[10][ports[1]] = attack[10].get(ports[1], 0) + 1

                if is_src and not is_dst:
                    if not switch_by_icmp:
                        attack[11][ip_address[0]] = attack[11].get(ip_address[0], 0) + 1
                        attack[12][ip_address[1]] = attack[12].get(ip_address[1], 0) + 1
                    else:
                        attack[11][ip_address[1]] = attack[11].get(ip_address[1], 0) + 1
                        attack[12][ip_address[0]] = attack[12].get(ip_address[0], 0) + 1
                if not is_src and is_dst:
                    if not switch_by_icmp:
                        attack[11][ip_address[1]] = attack[11].get(ip_address[1], 0) + 1
                        attack[12][ip_address[0]] = attack[12].get(ip_address[0], 0) + 1
                    else:
                        attack[11][ip_address[0]] = attack[11].get(ip_address[0], 0) + 1
                        attack[12][ip_address[1]] = attack[12].get(ip_address[1], 0) + 1
                if is_src and is_dst:
                    attack[11][ip_address[0]] = attack[11].get(ip_address[0], 0) + 1
                    attack[11][ip_address[1]] = attack[11].get(ip_address[1], 0) + 1
                if not is_src and not is_dst:
                    if ip_address[0]:
                        attack[13][ip_address[0]] = attack[13].get(ip_address[0], 0) + 1
                    if ip_address[1]:
                        attack[13][ip_address[1]] = attack[13].get(ip_address[1], 0) + 1
                statistics_attacks[k] = (attack[0],  # Label of attack
                                            attack[1] + 1,  # Packet count of attack
                                            attack[2] if attack[2] < float(
                                                d_point["meta.time_epoch"]) else float(
                                                d_point["meta.time_epoch"]),
                                            # Starttime
                                            attack[3] if attack[3] > float(
                                                d_point["meta.time_epoch"]) else float(
                                                d_point["meta.time_epoch"]),  # Endtime
                                            attack[4] + 1 if is_src else attack[4],
                                            # source Packets
                                            attack[5] + 1 if is_dst else attack[5],
                                            # Dest packets
                                            attack[
                                                6] + 1 if not is_src and not is_dst else
                                            attack[6],  # Other packets
                                            attack[7],  # protocols
                                            attack[8],  # Dict of Source Ports
                                            attack[9],  # Dict of Dest Ports
                                            attack[10],
                                            attack[11],
                                            attack[12],
                                            attack[13],
                                            attack[14] + int(
                                                d_point["meta.len"]) if is_src else
                                            attack[14],
                                            attack[15] + int(
                                                d_point["meta.len"]) if is_dst else
                                            attack[15],
                                            attack[16] + int(d_point[
                                                                 "meta.len"]) if not is_src and not is_dst else
                                            attack[16])  # Dict of other ports
                new_attack = False
                break

        if new_attack:
            statistics_attacks.append((
                cur_label,  # Label
                1,  # Packet count
                float(d_point["meta.time_epoch"]),  # Start time
                float(d_point["meta.time_epoch"]),  # End Time
                1 if is_src else 0,  # Source packets
                1 if is_dst else 0,  # Dest Packets
                1 if not is_src and not is_dst else 0,  # Other packets
                {d_point["meta.protocols"]: 1},  # Protocol
                {ports[0]: 1 if is_src or is_dst else 0},  # Source Ports
                {ports[1]: 1 if is_src or is_dst else 0},  # Dest Ports
                {ports[0]: 1 if not is_src and not is_dst else 0,
                 ports[1]: 1 if not is_src and not is_dst else 0},  # Other ports
                {ip_address[0]: 1 if (is_src and not switch_by_icmp) or (
                            is_dst and switch_by_icmp) or (is_src and is_dst) else 0,
                 ip_address[1]: 1 if (is_src and switch_by_icmp) or (
                             is_dst and not switch_by_icmp) or (
                                                 is_src and is_dst) else 0},
                {ip_address[0]: 1 if (is_src and switch_by_icmp) or (
                            is_dst and not switch_by_icmp) else 0,
                 ip_address[1]: 1 if (is_src and not switch_by_icmp) or (
                             is_dst and switch_by_icmp) else 0},
                {ip_address[0]: 1 if ip_address[0] and not is_src and not is_dst else 0,
                 ip_address[1]: 1 if ip_address[
                                         1] and not is_src and not is_dst else 0},
                int(d_point["meta.len"]) if is_src else 0,
                int(d_point["meta.len"]) if is_dst else 0,
                int(d_point["meta.len"]) if not is_src and not is_dst else 0
            ))

    if ip_address[0]:
        statistics_sips[ip_address[0]] = statistics_sips.get(ip_address[0], 0) + 1
    if ip_address[1]:
        statistics_dips[ip_address[1]] = statistics_dips.get(ip_address[1], 0) + 1

    if is_src and not is_dst:
        if not switch_by_icmp:
            statistics_ips[0][ip_address[0]] = statistics_ips[0].get(
                ip_address[0], 0) + 1
            statistics_ips[1][ip_address[1]] = statistics_ips[1].get(
                ip_address[1], 0) + 1
        else:
            statistics_ips[0][ip_address[1]] = statistics_ips[0].get(
                ip_address[1], 0) + 1
            statistics_ips[1][ip_address[0]] = statistics_ips[1].get(
                ip_address[0], 0) + 1
    if not is_src and is_dst:
        if not switch_by_icmp:
            statistics_ips[0][ip_address[1]] = statistics_ips[0].get(
                ip_address[1], 0) + 1
            statistics_ips[1][ip_address[0]] = statistics_ips[1].get(
                ip_address[0], 0) + 1
        else:
            statistics_ips[0][ip_address[0]] = statistics_ips[0].get(
                ip_address[0], 0) + 1
            statistics_ips[1][ip_address[1]] = statistics_ips[1].get(
                ip_address[1], 0) + 1
    if is_src and is_dst:
        statistics_ips[0][ip_address[0]] = statistics_ips[0].get(ip_address[0],
                                                                       0) + 1
        statistics_ips[0][ip_address[1]] = statistics_ips[0].get(ip_address[1],
                                                                       0) + 1
    if not is_src and not is_dst:
        if ip_address[0]:
            statistics_ips[2][ip_address[0]] = statistics_ips[2].get(
                ip_address[0], 0) + 1
        if ip_address[1]:
            statistics_ips[2][ip_address[1]] = statistics_ips[2].get(
                ip_address[1], 0) + 1

    if is_src or is_dst:
        if ports[0]:
            statistics_ports[0][ports[0]] = statistics_ports[0].get(ports[0],
                                                                          0) + 1
        if ports[1]:
            statistics_ports[1][ports[1]] = statistics_ports[1].get(ports[1],
                                                                          0) + 1
    else:
        if ports[0]:
            statistics_ports[2][ports[0]] = statistics_ports[2].get(ports[0],
                                                                          0) + 1
        if ports[1]:
            statistics_ports[2][ports[1]] = statistics_ports[2].get(ports[1],
                                                                          0) + 1

    return d_point


source = CSVFileDataSource(files="all_4.csv")
processor = DataProcessor().add_func(add_label).add_func(stats)
handler = DataHandler(data_source=source, data_processor=processor)
handler.open()
for p in handler:
    pass
handler.close()

with open("statistics", "w") as statistics:
    total_bytes = statistics_count[1] + statistics_count[2] + statistics_count[3]
    statistics.write(f"Packet Count: {statistics_count[0]}\n"
                     f"Packets send: {statistics_count[4]}  ({percent(statistics_count[4], statistics_count[0])}%)\n"
                     f"Packets received: {statistics_count[5]}  ({percent(statistics_count[5], statistics_count[0])}%)\n"
                     f"Packets with unknown origin: {statistics_count[6]}  ({percent(statistics_count[6], statistics_count[0])}%)\n"
                     f"Total Bytes: {total_bytes}\n"
                     f"Bytes send from Source: {statistics_count[1]}  ({percent(statistics_count[1], total_bytes)}%)\n"
                     f"Bytes send by Destination: {statistics_count[2]}  ({percent(statistics_count[2], total_bytes)}%)\n"
                     f"Bytes from unknown origin: {statistics_count[3]}  ({percent(statistics_count[3], total_bytes)}%)\n"
                     f"Protocols encountered:\n")
    remove_empty(statistics_count[7])
    total_prot = sum([statistics_count[7][x] for x in statistics_count[7]])
    for prot in statistics_count[7]:
        statistics.write(f"\t{prot}: {statistics_count[7][prot]}  ({percent(statistics_count[7][prot], total_prot)}%)\n")
    statistics.write(f"Labels encountered:\n")
    remove_empty(statistics_labels)
    total_label = sum([statistics_labels[x] for x in statistics_labels])
    for label in statistics_labels:
        statistics.write(f"\t{label}: {statistics_labels[label]}  ({percent(statistics_labels[label], total_label)}%)\n")
    statistics.write("Attacks encountered:\n")

    attack_p = {}
    attack_sp = {}
    attack_dp = {}
    attack_up = {}
    attack_sip = {}
    attack_dip = {}
    attack_uip = {}
    for attack in statistics_attacks:
        for key in attack[7]:
            attack_p[key] = attack_p.get(key, 0) + attack[7][key]
        for key in attack[8]:
            attack_sp[key] = attack_sp.get(key, 0) + attack[8][key]
        for key in attack[9]:
            attack_dp[key] = attack_dp.get(key, 0) + attack[9][key]
        for key in attack[10]:
            attack_up[key] = attack_up.get(key, 0) + attack[10][key]
        for key in attack[11]:
            attack_sip[key] = attack_sip.get(key, 0) + attack[11][key]
        for key in attack[12]:
            attack_dip[key] = attack_dip.get(key, 0) + attack[12][key]
        for key in attack[13]:
            attack_uip[key] = attack_uip.get(key, 0) + attack[13][key]
    try:
        total_attack = (
            "Attack Sum",
            sum([attack[1] for attack in statistics_attacks]),
            min([attack[2] for attack in statistics_attacks]),
            max([attack[3] for attack in statistics_attacks]),
            sum([attack[4] for attack in statistics_attacks]),
            sum([attack[5] for attack in statistics_attacks]),
            sum([attack[6] for attack in statistics_attacks]),
            attack_p,
            attack_sp,
            attack_dp,
            attack_up,
            attack_sip,
            attack_dip,
            attack_uip,
            sum([attack[14] for attack in statistics_attacks]),
            sum([attack[15] for attack in statistics_attacks]),
            sum([attack[16] for attack in statistics_attacks])
        )
    except ValueError:
        print("Error in attacks. There probably don't exist any.")
        total_attack = ("Attack Sum", 0, 0, 0, 0, 0, 0, attack_p, attack_sp, attack_dp, attack_up, attack_sip, attack_dip, attack_uip, 0, 0, 0)
    statistics_attacks.insert(0, total_attack)

    for attack in statistics_attacks:
        total_bytes = attack[14] + attack[15] + attack[16]
        statistics.write(f"\tAttack '{attack[0]}'\n"
                         f"\t\tPackets in Attack: {attack[1]}\n"
                         f"\t\tStarted at: {attack[2]}\t{str(datetime.datetime.fromtimestamp(attack[2]))}\n"
                         f"\t\tEnded at: {attack[3]}\t{str(datetime.datetime.fromtimestamp(attack[3]))}\n"
                         f"\t\tPackets send: {attack[4]}  ({percent(attack[4], attack[1])}%)\n"
                         f"\t\tPackets Received: {attack[5]}  ({percent(attack[5], attack[1])}%)\n"
                         f"\t\tPackets with unknown origin: {attack[6]}  ({percent(attack[6], attack[1])}%)\n"
                         f"\t\tBytes Total: {total_bytes}\n"
                         f"\t\tBytes send: {attack[14]}  ({percent(attack[14], total_bytes)}%)\n"
                         f"\t\tBytes received: {attack[15]}  ({percent(attack[15], total_bytes)}%)\n"
                         f"\t\tBytes unknown: {attack[16]}  ({percent(attack[16], total_bytes)}%)\n"
                         f"\t\tProtocols encountered:\n")
        remove_empty(attack[7])
        total_prot = sum([attack[7][x] for x in attack[7]])
        for prot in attack[7]:
            statistics.write(f"\t\t\t{prot}: {attack[7][prot]}  ({percent(attack[7][prot], total_prot)}%)\n")
        statistics.write(f"\t\tSource Ports:\n")
        remove_empty(attack[8])
        total_port = sum([attack[8][x] for x in attack[8]])
        for port in attack[8]:
            statistics.write(f"\t\t\t{port}: {attack[8][port]}  ({percent(attack[8][port], total_port)}%)\n")
        statistics.write("\t\tDestination Ports:\n")
        remove_empty(attack[9])
        total_port = sum([attack[9][x] for x in attack[9]])
        for port in attack[9]:
            statistics.write(f"\t\t\t{port}: {attack[9][port]}  ({percent(attack[9][port], total_port)}%)\n")
        statistics.write("\t\tPorts with unknown origin:\n")
        remove_empty(attack[10])
        total_port = sum([attack[10][x] for x in attack[10]])
        for port in attack[10]:
            statistics.write(f"\t\t\t{port}: {attack[10][port]}  ({percent(attack[10][port], total_port)}%)\n")
        statistics.write("\t\tOwn IPs:\n")
        remove_empty(attack[11])
        total_ip = sum([attack[11][x] for x in attack[11]])
        for ip in attack[11]:
            statistics.write(f"\t\t\t{ip}: {attack[11][ip]}  ({percent(attack[11][ip], total_ip)}%)\n")
        statistics.write("\t\tOthers IPs:\n")
        remove_empty(attack[12])
        total_ip = sum([attack[12][x] for x in attack[12]])
        for ip in attack[12]:
            statistics.write(f"\t\t\t{ip}: {attack[12][ip]}  ({percent(attack[12][ip], total_ip)}%)\n")
        statistics.write("\t\tUnknown Source IPs:\n")
        remove_empty(attack[13])
        total_ip = sum([attack[13][x] for x in attack[13]])
        for ip in attack[13]:
            statistics.write(f"\t\t\t{ip}: {attack[13][ip]}  ({percent(attack[13][ip], total_ip)}%)\n")
    statistics.write("Source IPs in Packets:\n")
    remove_empty(statistics_sips)
    total_ip = sum([statistics_sips[x] for x in statistics_sips])
    for ip in statistics_sips:
        statistics.write(f"\t{ip}: {statistics_sips[ip]}  ({percent(statistics_sips[ip], total_ip)}%)\n")
    statistics.write("Destination IPs in Packets:\n")
    remove_empty(statistics_dips)
    total_ip = sum([statistics_dips[x] for x in statistics_dips])
    for ip in statistics_dips:
        statistics.write(f"\t{ip}: {statistics_dips[ip]}  ({percent(statistics_dips[ip], total_ip)}%)\n")
    statistics.write("Source (own) IPs:\n")
    remove_empty(statistics_ips[0])
    total_ip = sum([statistics_ips[0][x] for x in statistics_ips[0]])
    for ip in statistics_ips[0]:
        statistics.write(f"\t{ip}: {statistics_ips[0][ip]}  ({percent(statistics_ips[0][ip], total_ip)}%)\n")
    statistics.write("Destination (others) IPs:\n")
    remove_empty(statistics_ips[1])
    total_ip = sum([statistics_ips[1][x] for x in statistics_ips[1]])
    for ip in statistics_ips[1]:
        statistics.write(f"\t{ip}: {statistics_ips[1][ip]}  ({percent(statistics_ips[1][ip], total_ip)}%)\n")
    statistics.write("IPs with unknown origin:\n")
    remove_empty(statistics_ips[2])
    total_ip = sum([statistics_ips[2][x] for x in statistics_ips[2]])
    for ip in statistics_ips[2]:
        statistics.write(f"\t{ip}: {statistics_ips[2][ip]}  ({percent(statistics_ips[2][ip], total_ip)}%)\n")
    statistics.write("Source Ports:\n")
    remove_empty(statistics_ports[0])
    total_port = sum([statistics_ports[0][x] for x in statistics_ports[0]])
    for port in statistics_ports[0]:
        statistics.write(f"\t{port}: {statistics_ports[0][port]}  ({percent(statistics_ports[0][port], total_port)}%)\n")
    statistics.write("Destination Ports:\n")
    remove_empty(statistics_ports[1])
    total_port = sum([statistics_ports[1][x] for x in statistics_ports[1]])
    for port in statistics_ports[1]:
        statistics.write(f"\t{port}: {statistics_ports[1][port]}  ({percent(statistics_ports[1][port], total_port)}%)\n")
    statistics.write("Ports with unknown origin:\n")
    remove_empty(statistics_ports[2])
    total_port = sum([statistics_ports[2][x] for x in statistics_ports[2]])
    for port in statistics_ports[2]:
        statistics.write(f"\t{port}: {statistics_ports[2][port]}  ({percent(statistics_ports[2][port], total_port)}%)\n")
    statistics.write("\n")

#relay = CSVFileRelay(target_file="processed.csv", overwrite_file=True, data_handler=handler, header_buffer_size=1000000, separator=";")
#relay.start(blocking=True)
#relay.stop()
