# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest


@pytest.fixture(scope="session")
def example_dict():
    d = {
        "header1": "data1",
        "header2": "data2",
        "header3": "data3",
        "header4": "data4",
        "header5": "data5",
        "header6": "data6",
        "header7": "data7",
        "header8": "data8",
    }
    return d
