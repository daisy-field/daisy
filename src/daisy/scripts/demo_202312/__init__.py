# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
    Pre-configured set of components of the federated IDS to perform the most simple demonstration of the entire system
    using a simple client-server topology, consisting of two federated IDS detection clients along the model aggregation
    server and two additional aggregation servers, one for the prediction results and a second one for the evaluation
    results.

    Each module of this package contains one of the following components, that can either be launched directly via the
    script (each containing a main) or via the command line, the name scheme of which consists of "demo_202312_NAME"
    with the following options:

        * client - Online IDS client processing one of the two splits of the demonstration dataset (see below).
        * server - TODO
        * evaluation_aggregator - TODO
        * prediction_aggregator - TODO

    The data used in this demonstration is the network traffic on Cohda boxes 2 and 5 on March 6th 2023 from the
    BeIntelli infrastructure, which must be available in (raw) pcap files, one path per device.
    Author: Fabian Hofmann
    Modified: 23.01.24
"""
from .client import create_client
