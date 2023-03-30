"""
    Parsing module for SPAT messages. Takes a parsed SPAT message (json) and extracts traffic light information from it.
    This includes the conversion of event_states into the actual TL colors and returning them along with the relevant
    info to which location (intersection id, signal group) it belongs to.

    Code is based on Diginet ITS code for the vehicle by Martin Berger (3.12.2019).

    Author: Fabian Hofmann, Martin Berger
    Modified: 10.8.22
"""

import json
import logging
import math
import time
from datetime import datetime as dt
from datetime import timedelta

from . import itstime


def get_intersection_id(intersection_state):
    """Retrieves the intersection ID from a given intersection state. Retrieval differs between versions due to the
    change in the message format (different/missing ids).

    @param intersection_state: intersection dict
    @return: unified intersection id
    """
    id_id = intersection_state["id"]["id"]

    if "region" not in intersection_state["id"]:
        id_region = "None"
    else:
        id_region = intersection_state["id"]["region"]

    return f"({id_region}-{id_id})"


def get_event_timing(movement_event, moy):
    """
    @param movement_event: movement event containing the start and end timestamp to be extracted
    @param moy: local time relative to the beginning of the current year
    @return: extracted start and end time of event
    """
    # if there is no time interval, make the event open-ended
    if 'timing' not in movement_event:
        return time.time(), math.inf  # FIXME could be computed by empirical means... not always possible if dynamic!

    timing = movement_event['timing']
    # start time is in newer versions not part of the field
    if "startTime" in timing:
        start_time = itstime.convert_spat_time(int(timing["startTime"]), moy)
    else:
        logging.debug(f"\t\tno start time found, falling back to current time")
        logging.warning(f"\t\tthis will cause problems with the demo samples since their timestamps are in the past!")
        start_time = time.time() # FIXME if multiple events are listed the start times are all the same!
    end_time = itstime.convert_spat_time(int(timing["minEndTime"]), moy)

    return start_time, end_time


def get_event_state_color(event_state):
    """
    @param event_state: string compliant with ETSI standard
    @return: TL color as string
    """
    if event_state in ["permissive-Movement-Allowed", "protected-Movement-Allowed"]:
        return "green"
    elif event_state in ["permissive-clearance", "protected-clearance"]:
        return "yellow"
    elif event_state in ["stop-Then-Proceed", "stop-And-Remain"]:
        return "red"
    elif event_state in ["pre-Movement"]:
        return "red-yellow"
    elif event_state in ["caution-Conflicting-Traffic"]:
        return "off"  # blinking yellow
    return "off"  # undefined


def check_timestamp(intersection_state):
    """Basic spat time stamp check for a given intersection state. Won't shut the program down in case the time stamps
    are way in the past or in the future, but complain about debugging purposes. As in, the spat messages won't be
    discarded.

    @param intersection_state: intersection dict
    """
    # local time stamp with beginning of year as ref point
    ts = timedelta(minutes=int(intersection_state["moy"]), seconds=-time.altzone,
                   milliseconds=int(intersection_state["timeStamp"]))

    spat_time = dt(dt.now().year, 1, 1) + ts
    logging.debug(f"\tspat timestamp: {spat_time}")

    msg_time = itstime.datetime_to_timestamp(spat_time)

    # age of message (sec), negative (future), positive (past)
    age = time.time() - msg_time
    if age < -60 * 60 * 24:
        logging.warning(f"spat message freshness (s): {age} < 1d -> future!")
    elif age < -10:
        logging.warning(f"spat message freshness (s): {age} < 10s -> future!")
    elif age < 0:
        logging.warning(f"spat message freshness (s): {age} < 0 -> future!")
    elif age > 24 * 60 * 60:
        logging.warning(f"spat message freshness (s): {age} > 1d -> too old!")
    elif age > 60 * 60:
        logging.warning(f"spat message freshness (s): {age} > 1h -> too old!")
    elif age > 10:
        logging.warning(f"spat message freshness (s): {age} > 10s -> too old!")


def handle_spatem(packet):
    spatem_p = packet["SPATEM"]

    if "header" not in spatem_p:
        raise ValueError("Unknown or no spatem header!")
    spatem_h = spatem_p["header"]
    logging.debug(f"SPATEM received: station_id={spatem_h['stationID']}, "
                  f"len={len(json.dumps(packet))}, time={dt.now()}")

    if "spat" not in spatem_p:
        raise ValueError("Unknown or no spat body!")
    spatem_b = spatem_p["spat"]

    # state of the entire intersection
    intersection_state = spatem_b["intersections"]["IntersectionState"]
    intersection_id = get_intersection_id(intersection_state)
    logging.debug(f"\tintersection id: {intersection_id}")

    # check freshness of message's timestamp
    moy = int(intersection_state["moy"])
    check_timestamp(intersection_state)

    # states of each group at the intersection
    movement_states = intersection_state["states"]["MovementState"]
    res = {}
    for movement_state in movement_states:
        signal_group_id = movement_state["signalGroup"]
        logging.debug(f"\t\tsignal group id: {signal_group_id}")

        # returns either several events (list) or a single elem (dict)
        movement_events = movement_state["state-time-speed"]["MovementEvent"]
        if type(movement_events) == dict:
            movement_events = [movement_events]

        event_list = []
        for movement_event in movement_events:
            # the event state is stored as a key with value == none
            event_state = list(movement_event['eventState'].keys())[0]
            logging.debug(f"\t\t[{signal_group_id}] event state: {event_state}")

            color = get_event_state_color(event_state)
            logging.debug(f"\t\t[{signal_group_id}] color: {color}")

            start_time, end_time = get_event_timing(movement_event, moy)
            logging.debug(f"\t\t[{signal_group_id}] start time {start_time} - end time {end_time}")

            event_list.append({'color': color,
                               'startTime': start_time,
                               'endTime': end_time})

        res[f"{intersection_id}-{signal_group_id}"] = event_list

    return res
