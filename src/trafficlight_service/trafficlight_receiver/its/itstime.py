"""
    Unchanged version of Diginet ITS package. Converts ITS timestamps for SPAT packets.

    Author: Martin Berger
    Modified: 3.12.2019
"""

import datetime
import time
from datetime import datetime as dt
from datetime import timedelta

import pytz

UTC = datetime.timezone.utc  # TODO rename to UTC_TZ


def datetime_to_timestamp(some_datetime):
    return time.mktime(some_datetime.timetuple())


def timestampIts_to_date(ts_its):
    offset = time.timezone
    # FIXME fishy, somehow daylight-savings ignored? cmp. time.altzone
    ref_tp = dt.fromtimestamp(ts_its / 1000. + time.mktime(dt(2004, 1, 1).timetuple()) - offset)
    return ref_tp


# def convert_spat_time(t, timestamp=None):  # direct transcript from java-script version.
#    """Converts a SPAT time into timestamp.
#    Parameter t is an offset to the UTC full hour with a resolution of 36 000 in units of 1/10th of second."""
#    timestamp = timestamp if timestamp else time.time()
#    full_hour = (int(timestamp) // 3600) * 3600.0
#    return full_hour + 0.1 * t


def convertSpatTime(t, minutesOfYear):
    """code copied from christian's (ITSMessageService.java (int,int):int))
    # unverified ..
    """
    startOfYear = dt(dt.now(UTC).year, 1, 1, 0, 0, 0, 0, UTC)
    startOfYear_in_s = datetime_to_timestamp(startOfYear)
    currentMinute = startOfYear_in_s + minutesOfYear * 60
    fullHour = currentMinute / 3600
    return fullHour * 3600.0 + 0.1 * t


def convert_spat_time(t, minuteOfYear=None):
    """adaptation of christian's version (from diginet-ps/ui project)
    """
    # unix timestamp of start of current year
    startOfYear = datetime_to_timestamp(dt(dt.now(UTC).year, 1, 1, 0, 0, 0, 0, UTC))
    tz_offset = datetime.datetime.now(pytz.timezone('Europe/Berlin')).utcoffset().total_seconds()
    if minuteOfYear is None:
        magic_offset = 0  # ?? ok, wow
        # magic_offset = 60 * 60  # FIXME off by 1 h that i CANNOT EXAPLAIN!
        minuteOfYear = int((time.time() - startOfYear - magic_offset) / 60)
        # print('  -- own moy:', minuteOfYear)
    currentMinute = startOfYear + minuteOfYear * 60  # current minute as timestamp
    fullHour = currentMinute // (60 * 60)  # integer division!

    fullHour += 1  # TODO ... magic correction value -.-

    return fullHour * (60 * 60) + 0.1 * t


def moy_to_full_hour(moy, seconds_since_full_hour):  # TODO fix misleading name!
    """adds 'seconds_since_full_hour' to 'minute_of_year' to get a full timestamp
    """
    dt_minute_acc = dt(dt.now(UTC).year, 1, 1, 0, 0, 0, 0, UTC) + timedelta(0, moy * 60)
    x = dt_minute_acc
    retval = dt(x.year, x.month, x.day, x.hour, 0, 0, 0, UTC) + timedelta(0, seconds_since_full_hour)
    return retval
