# Copyright (C) 2024 DAI-Labor and others
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

    * demo_202312_client - Basic F-IDS component demo using a simple client-server
    topology, along with the march23 dataset.

Author: Fabian Hofmann
Modified: 10.04.24
"""

__all__ = ["demo_202312_client", "demo_cic_client"]

from .demo_202312_client import create_client as demo_202312_client
from .demo_cic_client import create_client as demo_cic_client
