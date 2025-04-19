# Copyright (C) 2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from time import sleep

import keras
import numpy as np
import pytest

import threading

from daisy.data_sources import (
    DataHandler,
    PcapDataSource,
    PysharkProcessor,
    march23_event_handler,
    pcap_f_features,
    pcap_nn_aggregator,
)
from daisy.evaluation import ConfMatrSlidingWindowEvaluation
from daisy.federated_ids_components import (
    FederatedOnlineClient,
    FederatedModelAggregator,
    FederatedEvaluationAggregator,
    FederatedPredictionAggregator,
)
from daisy.federated_learning import (
    EMAvgTM,
    FederatedIFTM,
    TFFederatedModel,
    FedAvgAggregator,
)


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture
def simple_client():
    m_aggr_serv = ("localhost", 8003)
    eval_serv = ("localhost", 8001)
    aggr_serv = ("localhost", 8002)

    # Datasource
    source = PcapDataSource("../../../resources/pcap_v2x_test_file.pcap")
    processor = (
        PysharkProcessor()
        .packet_to_dict()
        .select_dict_features(features=pcap_f_features, default_value=np.nan)
        .merge_dict({"client_id": 2})
        .cast_dict_features(["meta.time_epoch", "ip.addr"], [float, str])
        .add_event_handler(march23_event_handler)
        .remove_dict_features(["client_id"])
        .dict_to_array(nn_aggregator=pcap_nn_aggregator)
    )
    data_handler = DataHandler(data_source=source, data_processor=processor)

    # Model
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss = keras.losses.MeanAbsoluteError()
    id_fn = TFFederatedModel.get_fae(
        input_size=44,
        optimizer=optimizer,
        loss=loss,
        batch_size=2,
        epochs=1,
    )
    t_m = EMAvgTM(alpha=0.05)
    err_fn = keras.losses.MeanAbsoluteError(reduction=None)
    model = FederatedIFTM(identify_fn=id_fn, threshold_m=t_m, error_fn=err_fn)

    metrics = [ConfMatrSlidingWindowEvaluation(window_size=2 * 2)]

    # Client
    return FederatedOnlineClient(
        data_handler=data_handler,
        batch_size=2 * 2,
        model=model,
        label_split=44,
        metrics=metrics,
        m_aggr_server=m_aggr_serv,
        eval_server=eval_serv,
        aggr_server=aggr_serv,
        update_interval_t=None,
    )


@pytest.fixture
def simple_m_aggr_server():
    return FederatedModelAggregator(
        m_aggr=FedAvgAggregator(),
        addr=("localhost", 8003),
        timeout=10,
        update_interval=None,
        num_clients=1,
        dashboard_url=None,
    )


@pytest.fixture
def simple_eval_server():
    return FederatedEvaluationAggregator(
        addr=("localhost", 8001), window_size=None, timeout=10, dashboard_url=None
    )


@pytest.fixture
def simple_aggr_server():
    return FederatedPredictionAggregator(
        addr=("localhost", 8002), window_size=None, timeout=10, dashboard_url=None
    )


class TestFederatedClientServer:
    def test_client_server(
        self,
        simple_client,
        simple_m_aggr_server,
        simple_eval_server,
        simple_aggr_server,
    ):
        def test_client():
            simple_client.start()
            sleep(20)
            simple_client.stop()

        def test_m_aggr_server():
            simple_m_aggr_server.start()
            sleep(20)
            simple_m_aggr_server.stop()

        def test_eval_server():
            simple_eval_server.start()
            sleep(20)
            simple_eval_server.stop()

        def test_aggr_server():
            simple_aggr_server.start()
            sleep(20)
            simple_aggr_server.stop()

        component_threads = [
            threading.Thread(target=test_client),
            threading.Thread(target=test_m_aggr_server),
            threading.Thread(target=test_eval_server),
            threading.Thread(target=test_aggr_server),
        ]

        for component_thread in component_threads:
            component_thread.start()

        for component_thread in component_threads:
            component_thread.join()
