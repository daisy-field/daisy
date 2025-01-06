# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Collection of various function to test packet handling of pyshark data sources.
These test-functions can be called directly, with the main on the bottom adjusted for
each test case. Each of them uses the same (direct) path to network traffic data from
the road-side infrastructure (BeIntelli) on Cohda boxes 2 on March 6th 2023,
which must be available in (raw) pcap files locally. Path should be adjusted as needed.

Author: Fabian Hofmann
Modified: 07.06.24
"""

import logging

from daisy.data_sources import (
    DataHandler,
    DataProcessor,
    packet_to_dict,
    select_feature,
    pcap_f_features,
    dict_to_numpy_array,
    pcap_nn_aggregator,
    CSVFileRelay,
    PcapDataSource,
    demo_202303_label_data_point,
)
import numpy as np

file_path = (
    "/home/fabian/Documents/DAI-Lab/DAISY/datasets"
    "/v2x_2023-03-06/diginet-cohda-box-dsrc2"
)


def pyshark_writer():
    """Creates and starts a file relay to write the contents of a pyshark file data
    source to a csv file that runs until processing all data points.
    """
    source = PcapDataSource(file_path)
    processor = (
        DataProcessor()
        .add_func(lambda o_point: packet_to_dict(o_point))
        .add_func(lambda o_point: select_feature(o_point, pcap_f_features, np.nan))
        .add_func(lambda o_point: demo_202303_label_data_point(2, o_point))
    )

    with DataHandler(
        data_source=source, data_processor=processor, multithreading=True
    ) as handler:
        relay = CSVFileRelay(
            data_handler=handler,
            target_file="v2x_A.csv",
            overwrite_file=True,
        )
        relay.start(blocking=True)


def pyshark_printer():
    """Creates and starts a data handler from a directory of pcap files,
    wrapped in a pcap data source and using the cohda processor to label it.
    """
    source = PcapDataSource(file_path)
    processor = (
        DataProcessor()
        .add_func(lambda o_point: packet_to_dict(o_point))
        .add_func(lambda o_point: select_feature(o_point, pcap_f_features, np.nan))
        .add_func(lambda o_point: demo_202303_label_data_point(2, o_point))
        .add_func(lambda o_point: dict_to_numpy_array(o_point, pcap_nn_aggregator))
    )

    with DataHandler(
        data_source=source, data_processor=processor, multithreading=True
    ) as handler:
        for item in handler:
            print(item)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)-10s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    pyshark_writer()
