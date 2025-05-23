# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Content used for the Dataset Demo from March 6th 2023.

Author: Jonathan Ackerschewski, Fabian Hofmann
Modified: 04.11.2024
"""

from datetime import datetime, timezone, timedelta

from typing_extensions import deprecated

from ..events import EventHandler

# Existing datasets captured on Cohda boxes 2 and 5 on March 6th (2023)
# contains attacks in the following:
# 1: "Installation Attack Tool"
# 2: "SSH Brute Force"
# 3: "SSH Privilege Escalation"
# 4: "SSH Brute Force Response"
# 5: "SSH Data Leakage"
march23_event_handler = (
    EventHandler(default_label="0")
    .append_event(
        "1",
        "meta.time_epoch > t!06.03.23T12:34:17+01:00 and "
        "meta.time_epoch < t!06.03.23T12:40:28+01:00 and "
        "(client_id = i!5 and (http in meta.protocols or tcp in meta.protocols) and "
        "192.168.213.86 in ip.addr and 185. in ip.addr)",
    )
    .append_event(
        "2",
        "meta.time_epoch > t!06.03.23T12:49:04+01:00 and"
        "meta.time_epoch < t!06.03.23T13:23:16+01:00 and"
        "(client_id = i!5 and (ssh in meta.protocols or tcp in meta.protocols) and "
        "192.168.230.3 in ip.addr and 192.168.213.86 in ip.addr)",
    )
    .append_event(
        "3",
        "meta.time_epoch > t!06.03.23T13:25:27+01:00 and"
        "meta.time_epoch < t!06.03.23T13:31:11+01:00 and"
        "(client_id = i!5 and (ssh in meta.protocols or tcp in meta.protocols) and "
        "192.168.230.3 in ip.addr and 192.168.213.86 in ip.addr)",
    )
    .append_event(
        "4",
        "meta.time_epoch > t!06.03.23T12:49:04+01:00 and"
        "meta.time_epoch < t!06.03.23T13:23:16+01:00 and"
        "(client_id = i!2 and (ssh in meta.protocols or tcp in meta.protocols) and "
        "192.168.230.3 in ip.addr and 130.149.98.119 in ip.addr)",
    )
    .append_event(
        "5",
        "meta.time_epoch > t!06.03.23T13:25:27+01:00 and"
        "meta.time_epoch < t!06.03.23T13:31:11+01:00 and"
        "(client_id = i!2 and (ssh in meta.protocols or tcp in meta.protocols) and "
        "192.168.230.3 in ip.addr and 130.149.98.119 in ip.addr)",
    )
)

_deprecated_march23_event_handler = (
    EventHandler(default_label="0")
    .add_event(
        datetime(2023, 3, 6, 12, 34, 17, tzinfo=timezone(timedelta(hours=1))),
        datetime(2023, 3, 6, 12, 40, 28, tzinfo=timezone(timedelta(hours=1))),
        "1",
        "client_id = 5 and (http in meta.protocols or tcp in meta.protocols) and 192.168.213.86 in ip.addr and 185. in ip.addr",
    )
    .add_event(
        datetime(2023, 3, 6, 12, 49, 4, tzinfo=timezone(timedelta(hours=1))),
        datetime(2023, 3, 6, 13, 23, 16, tzinfo=timezone(timedelta(hours=1))),
        "2",
        "client_id = 5 and (ssh in meta.protocols or tcp in meta.protocols) and 192.168.230.3 in ip.addr and 192.168.213.86 in ip.addr",
    )
    .add_event(
        datetime(2023, 3, 6, 13, 25, 27, tzinfo=timezone(timedelta(hours=1))),
        datetime(2023, 3, 6, 13, 31, 11, tzinfo=timezone(timedelta(hours=1))),
        "3",
        "client_id = 5 and (ssh in meta.protocols or tcp in meta.protocols) and 192.168.230.3 in ip.addr and 192.168.213.86 in ip.addr",
    )
    .add_event(
        datetime(2023, 3, 6, 12, 49, 4, tzinfo=timezone(timedelta(hours=1))),
        datetime(2023, 3, 6, 13, 23, 16, tzinfo=timezone(timedelta(hours=1))),
        "4",
        "client_id = 2 and (ssh in meta.protocols or tcp in meta.protocols) and 192.168.230.3 in ip.addr and 130.149.98.119 in ip.addr",
    )
    .add_event(
        datetime(2023, 3, 6, 13, 25, 27, tzinfo=timezone(timedelta(hours=1))),
        datetime(2023, 3, 6, 13, 31, 11, tzinfo=timezone(timedelta(hours=1))),
        "5",
        "client_id = 2 and (ssh in meta.protocols or tcp in meta.protocols) and 192.168.230.3 in ip.addr and 130.149.98.119 in ip.addr",
    )
)


@deprecated("Use the Data Processor instead")
def demo_202303_label_data_point(client_id: int, d_point: dict) -> dict:
    """Labels the data points according to the events for the demo 202303.

    :param client_id: Client ID.
    :param d_point: Data point as dictionary.
    :return: Labeled data point.
    """
    return _deprecated_march23_event_handler.process(
        datetime.fromtimestamp(float(d_point["meta.time_epoch"])),
        d_point,
        [{"client_id": str(client_id)}],
    )
