# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Collection of various tests of the data source package. These tests are merely for
development purposes only and are not unit-test compliant. Due to the nature of
multi-threading, these tests may or may not be deterministic as well in their way of
running through the procedures, merely in their final output!

Currently, the following test-modules are provided:

    * simple_handler - Data handler and relay test functions using simple data points.
    * pyshark_handler - Data handler and relays test functions using pyshark data
    points.

Author: Fabian Hofmann
Modified: 06.06.24
"""
