# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pyshark

py = pyshark.LiveCapture(interface="any")
gen = py.sniff_continuously()

for packet in gen:
    print(f"got packet: {packet}")
