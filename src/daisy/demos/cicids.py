# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Content used for the Dataset Demo from March 6th 2023.

Author: Jonathan Ackerschewski, Seraphin Zunzer
Modified: 04.11.2024
"""


def cic_label_data_point(d_point: dict) -> dict:
    """removes the original label key and labels the data points coorectly.

    :param d_point: Data point as dictionary.
    :return: Labeled data point.
    """
    label = d_point.pop(" Label")
    d_point["label"] = 0 if label == "BENIGN" else 1
    return d_point
