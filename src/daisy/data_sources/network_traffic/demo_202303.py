# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Content used for the Dataset Demo from March 6th 2023.

Author: Jonathan Ackerschewski, Fabian Hofmann
Modified: 04.11.2024
"""

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
    .add_event(
        "1",
        "meta.time_epoch > t!06.03.23T12:34:17+01:00 and "
        "meta.time_epoch < t!06.03.23T12:40:28+01:00 and "
        "(client_id = i!5 and (http in meta.protocols or tcp in meta.protocols) and "
        "192.168.213.86 in ip.addr and 185. in ip.addr)",
    )
    .add_event(
        "2",
        "meta.time_epoch > t!06.03.23T12:49:04+01:00 and"
        "meta.time_epoch < t!06.03.23T13:23:16+01:00 and"
        "(client_id = i!5 and (ssh in meta.protocols or tcp in meta.protocols) and "
        "192.168.230.3 in ip.addr and 192.168.213.86 in ip.addr)",
    )
    .add_event(
        "3",
        "meta.time_epoch > t!06.03.23T13:25:27+01:00 and"
        "meta.time_epoch < t!06.03.23T13:31:11+01:00 and"
        "(client_id = i!5 and (ssh in meta.protocols or tcp in meta.protocols) and "
        "192.168.230.3 in ip.addr and 192.168.213.86 in ip.addr)",
    )
    .add_event(
        "4",
        "meta.time_epoch > t!06.03.23T12:49:04+01:00 and"
        "meta.time_epoch < t!06.03.23T13:23:16+01:00 and"
        "(client_id = i!2 and (ssh in meta.protocols or tcp in meta.protocols) and "
        "192.168.230.3 in ip.addr and 130.149.98.119 in ip.addr)",
    )
    .add_event(
        "5",
        "meta.time_epoch > t!06.03.23T13:25:27+01:00 and"
        "meta.time_epoch < t!06.03.23T13:31:11+01:00 and"
        "(client_id = i!2 and (ssh in meta.protocols or tcp in meta.protocols) and "
        "192.168.230.3 in ip.addr and 130.149.98.119 in ip.addr)",
    )
)
