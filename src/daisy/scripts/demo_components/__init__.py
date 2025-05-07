# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Collection of executable scripts to start up the various components of the daisy
package, more or less pre-configured by the respective script for ease of use and
demonstration purposes. Each module contains a demo of a single component,
i.e. script to be started, either in separate python instances or via threads.
Alternatively one can also launch these scripts directly via the command line. Often,
for a full demo, one requires multiple scripts, as demo setups may require generic
components as well. See the docstrings of the respective demos.

Currently, the following script-demos are provided:

    * demo_202303_client - Basic F-IDS component demo using a simple client-server
    topology, along with the march23 dataset.
    * TODO

Author: Fabian Hofmann
Modified: 10.04.24
"""

__all__ = [
    "v2x_2023_03_06_client",
    "demo_jammer",
]

from .v2x_2023_03_06_client import create_client as v2x_2023_03_06_client
from .demo_jammer import create_relay as demo_jammer
