"""
    TODO

    Author: Seraphin Zunzer, Fabian Hofmann
    Modified: 02.05.23
"""

from data_sources.network_traffic.traffic_source import TrafficSource
from datetime import datetime


# TODO: Differentiate between client IDS
# Can i assume that the default filter ist valid.


class CohdaSource(TrafficSource):
    """ A class inheriting from the Traffic Source and overwriting the MAP function.
        Used to process the existing datasets captured on COHDA Boxes 2 and 5 on March 6th.
        Labels are appended according to the used protocol, timestamps, source and destination ip addresses.
    """

    def compare(self, point_ts: datetime, attack_ts: []):
        """Check if date time object from a datapoint is in between two date time objects

        :param attack_ts: array of two datetime objects [start, end]
        :param point_ts: datetime object for datapoint
        :return: boolean: true if datapoint time is in time range
        """
        return attack_ts[0] <= point_ts <= attack_ts[1]

    def map(self, o_point: dict):
        """Add labels to datapoint based on the simulated attacks in the collected network data

        :param o_point: dictionary of datapoint
        :return: dictionary including label
        """

        attack_tool_ts = [datetime(2023, 3, 6, 12, 34, 00), datetime(2023, 3, 6, 12, 40, 00)]
        brute_force_ts = [datetime(2023, 3, 6, 12, 49, 00), datetime(2023, 3, 6, 13, 23, 00)]
        privilege_ts = [datetime(2023, 3, 6, 13, 25, 00), datetime(2023, 3, 6, 13, 32, 00)]

        if (CLIENT_ID == 5 and self.compare(o_point['meta.time'].to_datetime(), attack_tool_ts)
                and any([x in o_point['meta.protocols'] for x in ["http", "tcp"]])
                and all([x in o_point['ip.addr'] for x in ["192.168.213.86", "185."]])):
            o_point['label'] = "Installation Attack Tool"

        elif (CLIENT_ID == 5 and self.compare(o_point['meta.time'].to_datetime(), brute_force_ts)
              and any([x in o_point['meta.protocols'] for x in ["ssh", "tcp"]])
              and all([x in o_point['ip.addr'] for x in ["192.168.230.3", "192.168.213.86"]])):
            o_point['label'] = "SSH Brute Force"

        elif (CLIENT_ID == 5 and self.compare(o_point['meta.time'].to_datetime(), privilege_ts)
              and any([x in o_point['meta.protocols'] for x in ["ssh", "tcp"]])
              and all([x in o_point['ip.addr'] for x in ["192.168.230.3", "192.168.213.86"]])):
            o_point['label'] = "SSH Privilege Escalation"

        elif (CLIENT_ID == 2 and self.compare(o_point['meta.time'].to_datetime(), brute_force_ts)
              and any([x in o_point['meta.protocols'] for x in ["ssh", "tcp"]])
              and all([x in o_point['ip.addr'] for x in ["192.168.230.3", "130.149.98.119"]])):
            o_point['label'] = "SSH Brute Force Response"

        elif (CLIENT_ID == 2 and self.compare(o_point['meta.time'].to_datetime(), privilege_ts)
              and any([x in o_point['meta.protocols'] for x in ["ssh", "tcp"]])
              and all([x in o_point['ip.addr'] for x in ["192.168.230.3", "130.149.98.119"]])):
            o_point['label'] = "SSH Data Leakage"

        else:
            o_point['label'] = "Normal"

        print("Map applied")
        return o_point
