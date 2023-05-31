from data_sources.network_traffic.traffic_source import TrafficSource
from datetime import datetime
from typing import Tuple


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


# TODO: Differentiate between client IDS
# Can i assume that the default filter ist valid.


class CohdaSource(TrafficSource):
    """ A class inheriting from the Traffic Source and overwriting the MAP function.
        Used to process the existing datasets captured on COHDA Boxes 2 and 5 on March 6th.
        Labels are appended according to the used protocol, timestamps, source and destination ip addresses.
    """

    def compare_times(self, point_ts: datetime, attack_ts: []):
        """
            Check if date time object from a datapoint is in between two date time objects

            :param attack_ts: array of two datetime objects [start, end]
            :param point_ts: datetime object for datapoint
            :return: boolean: true if datapoint time is in time range
        """
        return attack_ts[0] <= point_ts <= attack_ts[1]

    def filter_anomaly(self, client_id: int, client_target: int, attack_time: Tuple[datetime, datetime], target_protocols: [],
                target_ip_addresses: [], label: string, data_point:[]):
        """Check if datapoint is an anomaly and add label if condition is fulfilled

        :param client_id: Client ID of the client this cohda sorce sends the traffic to
        :param client_target: The ID of the client that was affected of this attack
        :param target_protocols: Protocols that were targeted
        :param start_time: Start time of the attack
        :param end_time: End time of the attack
        :param target_ip_addresses: List of IP addresses that were targeted
        :param label: String that should be the label
        :param data_point: Datapoint to test for anomaly
        :return: data_point dict with added label if anomaly conditions are fulfilled
        """
        if data_point.get("label") is None and \
                    (client_id == client_target and
                    self.compare_times(data_point['meta.time'].to_datetime(), [attack_time[0], attack_time[1]]) and
                    any([x in datapoint['meta.protocols'] for x in target_protocols]) and
                    all([x in o_point['ip.addr'] for x in target_ip_addresses])):
                data_point['label'] = label
                return data_point



    def map_anomalies(self, o_point: dict):
        """Add labels to datapoint based on the simulated attacks in the collected network data

        :param o_point: dictionary of datapoint
        :return: dictionary including label
        """

        attack_tool_ts = Tuple[datetime(2023, 3, 6, 12, 34, 00), datetime(2023, 3, 6, 12, 40, 00)]
        brute_force_ts = Tuple[datetime(2023, 3, 6, 12, 49, 00), datetime(2023, 3, 6, 13, 23, 00)]
        privilege_ts = Tuple[datetime(2023, 3, 6, 13, 25, 00), datetime(2023, 3, 6, 13, 32, 00)]

        self.label_anomalies(CLIENT_ID, 5, attack_tool_ts,
                             ["http","tcp"], ["192.168.213.86", "185."], "Installation Attack Tool")
        self.label_anomalies(CLIENT_ID, 5, brute_force_ts,
                             ["ssh", "tcp"], ["192.168.230.3", "192.168.213.86"], "SSH Brute Force")
        self.label_anomalies(CLIENT_ID, 5, privilege_ts,
                             ["ssh", "tcp"], ["192.168.230.3", "192.168.213.86"], "SSH Privilege Escalation")

        self.label_anomalies(CLIENT_ID, 2, brute_force_ts,
                             ["ssh", "tcp"], ["192.168.230.3", "130.149.98.119"], "SSH Brute Force Response")
        self.label_anomalies(CLIENT_ID, 2, privilege_ts,
                             ["ssh", "tcp"], ["192.168.230.3", "130.149.98.119"], "SSH Data Leakage")

        if o_point.get("label") is None:
            o_point['label'] = "Normal"

        logging.info("Map applied")
        return o_point
