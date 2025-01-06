# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Collection of executable scripts to start data collection components that
individually monitor, pre-process (even label), and store data from pre-defined types
of data sources, more or less pre-configured by the respective script for ease
of use and demonstration purposes. Each module contains the setup of a single specific
data collector for a specific type of data, i.e. script to be started, either in
separate python instances or via threads. Alternatively one can also launch these
scripts directly via the command line.

Currently, the following (sub-)packaged scripts are provided:

    * Pyshark data collector - Collects live network traffic using pyshark, processes
    it and stores it in a CSV file.

Author: Jonathan Ackerschewski, Fabian Hofmann
Modified: 25.10.2024
"""

__all__ = ["pyshark_data_collector"]

from .pyshark_data_collector import create_collector as pyshark_data_collector
