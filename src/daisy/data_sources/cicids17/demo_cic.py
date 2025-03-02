# Copyright (C) 2024 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Content used for the Dataset Demo from 12.2023.

Author: Jonathan Ackerschewski, Fabian Hofmann
Modified: 18.10.2024
"""


def cic_label_data_point(client_id: int, d_point: dict) -> dict:
    """Labels the data points according to the events for the demo 202312

    :param client_id: THe client ID
    :param d_point: Data point as dictionary
    :return: Labeled data point
    """
    label = d_point.pop(" Label")
    if label == "BENIGN":
        d_point["label"] = 0
    else:
        d_point["label"] = 1

    return d_point
