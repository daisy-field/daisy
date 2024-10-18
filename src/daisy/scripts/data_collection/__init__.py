# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""This package contains scripts for collecting traffic.

    * Pyshark data collector - collects live network traffic using pyshark, processes
        it and stores it in a CSV file.

Author: Jonathan Ackerschewski
Modified: 11.10.2024

"""

__all__ = ["pyshark_data_collector"]

from .pyshark_data_collector import create_collector as pyshark_data_collector
