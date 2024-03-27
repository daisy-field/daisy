# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""TODO
Collection of executable scripts to start up the various components of the package, more or less pre-configured by
the respective script for ease of use and demonstration purposes. Each subpackage contains a bundled demo, i.e. a
set of scripts to be started, either in separate python instances or via threads. Alternatively one can also launch
these scripts directly via the command line. Currently, the following (sub-)packaged script-demos are provided:

    * demo_202312 - Basic F-IDS component demo using a client-server topology, along with the march23 dataset.

Author: Fabian Hofmann
Modified: 27.02.24




Pre-configured set of components of the federated IDS to perform the most simple demonstration of the entire system
using a simple client-server topology, consisting of two federated IDS detection clients along the model aggregation
server and two additional aggregation servers, one for the prediction results and a second one for the evaluation
results.

Each module of this package contains one of the following components, that can either be launched directly via the
script (each containing a main) or via the command line, the name scheme of which consists of "demo_202312_NAME"
with the following options:

    * client - Online IDS client processing one of the two splits of the demonstration dataset (see below).
    * server - TODO
    * eval_aggr_server - TODO
    * pred_aggr_server - TODO


Author: Fabian Hofmann
Modified: 27.02.24
"""

from .demo_202312_client import create_client as demo_202312_client
