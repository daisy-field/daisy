# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Collection of executable scripts to start up the various components of the daisy
package (as a framework), more or less pre-configured by the respective script for
ease of use. Some of the scripts and entrypoints inside the respective subpackages
function as generic components (see below) served by the framework, while other are
more narrow configurations/implementations of said components to be run a demo
examples to work as a functioning setup. Most likely, for a full demo, one requires
scripts from different packages, as demo setups may require generic components as
well. See the docstrings of the respective demos. No matter the subpackage,
the scripts can be started either in separate python instances or via threads. For
convenience, one can also launch these scripts directly via the command line.

Currently, the following (sub-)packaged scripts are provided:

    * data_collection - Pre-configured scripts to label, process and store monitoring
    data from pre-selected typs of data sources.
    * demo_components - Pre-configured demo components for specific inputs/scenarios.
    * generic_fids_components - General federated IDS components required for
    different (demo-)scenarios

Author: Fabian Hofmann, Seraphin Zunzer, Jonathan Ackerschewski
Modified: 25.10.24
"""
