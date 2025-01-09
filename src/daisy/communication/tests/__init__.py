# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Collection of various tests of the message stream module for endpoints and
endpoint server. These tests are merely for development purposes only and are not
unit-test compliant. Due to the nature of multi-threading, these tests may or may not
be deterministic as well.

Currently, the following test-modules are provided:

    * simple_acceptor - Acceptor endpoint (server) side test functions.
    * simple_initiator - Initiator endpoint (client) side test functions.
    * simple_server - Dedicated endpoint server function to create test servers.

To start any tests, initiator(s) must always be launched in tandem with either
acceptors(s) or a server from their respective modules. Some acceptor tests can also
be launched in standalone manner. See their modules' and their respective functions'
docstrings for more information.

Author: Fabian Hofmann
Modified: 10.04.24
"""
