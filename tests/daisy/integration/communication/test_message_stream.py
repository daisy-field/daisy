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
def simple_initiator(request):
    initiator = StreamEndpoint(
        name="Initiator",
        addr=("127.0.0.1", 13000),
        remote_addr=("127.0.0.1", 32000),
        acceptor=False,
        multithreading=request.param,
    )
    return initiator


@pytest.fixture(scope="function")
def simple_acceptor(request):
    acceptor = StreamEndpoint(
        name="Acceptor",
        addr=("127.0.0.1", 32000),
        remote_addr=("127.0.0.1", 13000),
        acceptor=True,
        multithreading=request.param,
    )
    return acceptor


class TestStreamEndpoint:
    @pytest.mark.slow_integration_test
    @pytest.mark.parametrize(
        "simple_initiator, simple_acceptor",
        [
            (True, True),
            (False, False),
            (True, False),
            (False, True),
        ],
        indirect=["simple_initiator", "simple_acceptor"],
    )
    def test_send_recv(
        self,
        simple_initiator: StreamEndpoint,
        simple_acceptor: StreamEndpoint,
        example_list: list,
    ):
        initiator_rdy = simple_initiator.start()
        acceptor_rdy = simple_acceptor.start()
        initiator_rdy.wait()
        acceptor_rdy.wait()

        for element in example_list:
            simple_initiator.send(element)
            simple_acceptor.send(element)

        initiator_recv_list = []
        acceptor_recv_list = []
        for _ in range(5):
            initiator_recv_list.append(simple_initiator.receive())
            acceptor_recv_list.append(simple_acceptor.receive())

        assert initiator_recv_list == acceptor_recv_list == example_list

        initiator_stp = simple_initiator.stop(shutdown=True)
        acceptor_stp = simple_acceptor.stop(shutdown=True)
        initiator_stp.wait()
        acceptor_stp.wait()
