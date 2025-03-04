# Copyright (C) 2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest

from daisy.communication import StreamEndpoint


@pytest.fixture
def example_list():
    return [1, 2, 3, 4, 5]


@pytest.fixture(scope="function")
def simple_initiator(multithreading):
    return StreamEndpoint(
        name="Initiator",
        addr=("127.0.0.1", 13000),
        remote_addr=("127.0.0.1", 32000),
        acceptor=False,
        multithreading=multithreading.param,
    )


@pytest.fixture(scope="function")
def simple_acceptor(multithreading):
    return StreamEndpoint(
        name="Acceptor",
        addr=("127.0.0.1", 32000),
        remote_addr=("127.0.0.1", 13000),
        acceptor=True,
        multithreading=multithreading.param,
    )


# @pytest.fixture(scope="function")


class TestStreamEndpoint:
    @pytest.mark.parametrize(
        "simple_initiator,simple_acceptor",
        [
            (True, True),
            (False, False),
            (True, False),
            (False, True),
        ],
        indirect=True,
    )
    def test_acceptor(self, simple_initiator, simple_acceptor):
        print(simple_initiator._multithreading, simple_acceptor._multithreading)
