# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Collection of executable scripts to start up the various federated IDS components
of the daisy package, more or less pre-configured by the respective script for ease
of use and demonstration purposes. Each module contains the setup of a single generic
F-IDS component, i.e. script to be started, either in separate python instances or
via threads. Alternatively one can also launch these scripts directly via the command
line. Most of these components require other components however to form a functioning
systems, e.g. demo setups require specific demo components as well. See the
docstrings of the respective demos.

Currently, the following generic federated IDS components are provided:

    * dashboard - Auxiliary dashboard for all other servers' aggregated results.
    * eval_aggr_server - Auxiliary aggregation server for evaluation metric results.
    * model_aggr_server - Federated model aggregation server using federated averaging.
    * pred_aggr_server - Auxiliary aggregation server for prediction value results.

Author: Fabian Hofmann
Modified: 17.06.24
"""

__all__ = ["dashboard", "eval_aggr_server", "model_aggr_server", "pred_aggr_server"]

from .dashboard import create_dashboard as dashboard
from .eval_aggr_server import create_server as eval_aggr_server
from .model_aggr_server import create_server as model_aggr_server
from .pred_aggr_server import create_server as pred_aggr_server
