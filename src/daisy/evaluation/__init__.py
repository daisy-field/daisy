# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Toolbox for evaluation for the performance of federated online learning approaches,
namely for intrusion detection. Since they require various kinds of inputs,
there is no singular interface to implement and call when using one of them,
check the individual modules for more info.

At the moment, the toolbox contains a set of classe for the online evaluation of
anomaly detection approaches using customized extensions of the keras metric API:

    * SlidingWindowEvaluation - Interface class. Implementations must compute
    incremental updates to the metric.
    * ConfMatrSlidingWindowEvaluation - Evaluator that computes the confusion matrix
    metrics over a sliding window.
    * TFMetricSlidingWindowEvaluation - Class wrapper for generic tensorflow metrics
    over a sliding window.

Author: Fabian Hofmann
Modified: 03.04.24
"""

__all__ = [
    "SlidingWindowEvaluation",
    "ConfMatrSlidingWindowEvaluation",
    "TFMetricSlidingWindowEvaluation",
]

from .anomaly_detection_online_evaluation import (
    ConfMatrSlidingWindowEvaluation,
    TFMetricSlidingWindowEvaluation,
)
from .anomaly_detection_online_evaluation import SlidingWindowEvaluation
