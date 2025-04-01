# Copyright (C) 2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import logging
import threading

import pytest

from daisy.communication import StreamEndpoint


@pytest.fixture(autouse=True)
def reset():
    yield

    # noinspection PyProtectedMember
    from daisy.communication.message_stream import _EndpointSocket

    _EndpointSocket._listen_socks = {}
    _EndpointSocket._acc_r_socks = {}
    _EndpointSocket._acc_p_socks = {}
    _EndpointSocket._reg_r_addrs = set()
    _EndpointSocket._addr_map = {}
    _EndpointSocket._act_l_counts = {}
    _EndpointSocket._lock = threading.Lock()
    _EndpointSocket._cls_logger = logging.getLogger("EndpointSocketCLS")
    _EndpointSocket._cls_log_level = None


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
    def test_ping_pong(
        self,
        simple_initiator: StreamEndpoint,
        simple_acceptor: StreamEndpoint,
    ):
        """Creates and starts an acceptor to perform five ping-pong tests with the
        opposing initiator, sending out "pong" and receiving "ping" messages.
        """
        initiator_rdy = simple_initiator.start()
        acceptor_rdy = simple_acceptor.start()
        initiator_rdy.wait()
        acceptor_rdy.wait()

        for _ in range(5):
            simple_initiator.send("ping")
            msg = simple_acceptor.receive()
            assert msg == "ping"
            simple_acceptor.send("pong")
            msg = simple_initiator.receive()
            assert msg == "pong"

        initiator_stp = simple_initiator.stop(shutdown=True)
        acceptor_stp = simple_acceptor.stop(shutdown=True)
        initiator_stp.wait()
        acceptor_stp.wait()

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
        """Creates and starts an acceptor-initiator pair to send and receive five
        messages each.
        """
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

    @pytest.mark.integration_test
    def test_clashing_acceptor(
        self,
    ):
        """Creates multiple acceptor endpoints that have the same address (which is
        supported by the underlying endpoint sockets) but also the same
        remote (initiator) address which should result in a double registration
        causing an error.
        """
        acceptor_1 = StreamEndpoint(
            name=f"Acceptor-{1}",
            addr=("127.0.0.1", 13000),
            remote_addr=("127.0.0.1", 32000),
            acceptor=True,
        )
        with pytest.raises(ValueError):
            StreamEndpoint(
                name=f"Acceptor-{2}",
                addr=("127.0.0.1", 13000),
                remote_addr=("127.0.0.1", 32000),
                acceptor=True,
            )
        acceptor_1.stop(shutdown=True)

    @pytest.mark.slow_integration_test
    @pytest.mark.parametrize(
        "simple_acceptor",
        [
            True,
            False,
        ],
        indirect=["simple_acceptor"],
    )
    def test_one_time_initiator(
        self,
        simple_acceptor: StreamEndpoint,
        example_list: list,
    ):
        """Creates and starts an initiator to send out multiple messages before
        stopping the endpoint, all of which done through the helper class method of the
        endpoint class.
        """
        StreamEndpoint.create_quick_sender_ep(
            example_list,
            addr=("127.0.0.1", 13000),
            remote_addr=("127.0.0.1", 32000),
            blocking=False,
        )
        acceptor_rdy = simple_acceptor.start()
        acceptor_rdy.wait()

        acceptor_recv_list = []
        for _ in range(5):
            try:
                acceptor_recv_list.append(simple_acceptor.receive(timeout=10))
            except TimeoutError:
                assert False, "Receive took too long, message must have been lost!"
        assert acceptor_recv_list == example_list

        acceptor_stp = simple_acceptor.stop(shutdown=True)
        acceptor_stp.wait()
