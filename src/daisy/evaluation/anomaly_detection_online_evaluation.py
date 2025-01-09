# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Extensions to the tensorflow metric set, to measure the quality of online anomaly
detection approaches. For now, only a sliding-window-based solution is realized,
as this works for any kind of metric (not every metric can be computed online).

Author: Fabian Hofmann
Modified: 17.08.23
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Self

import tensorflow as tf
from tensorflow import keras, Tensor


class SlidingWindowEvaluation(keras.metrics.Metric, ABC):
    """Abstract evaluation metric class that extends the existing tensorflow metric
    base class, with a sliding window to collect the k most recent predicted labels
    to evaluate the model's recent performance on them in point-wise manner.

    Note that depending on the metric, non-abstract methods must be extended with a
    new metric's own functionality.
    """

    true_labels: deque
    pred_labels: deque
    _window_size: int

    def __init__(self, name="ad_online_evaluation", window_size: int = None, **kwargs):
        """Creates a new sliding window evaluation metric.

        :param name: Name of metric.
        :param window_size: Size of sliding window. If not provided, assume infinite
        window size.
        :param kwargs: Additional metric/layer keywords arguments.
        """
        super().__init__(name=name, **kwargs)

        self.true_labels = deque()
        self.pred_labels = deque()
        self._window_size = window_size

    def update_state(self, y_true, y_pred, *args, **kwargs):
        """Adds a mini-batch of inputs to the metric, removing old ones if the window
        is full, and adjusting statistics accordingly. Converts any tensors into
        numpy arrays as data points/pairs are processed in element-wise fashion
        anyway and this makes it easier for generic handling.

        :param y_true: Vector/Tensor containing true labels of inputs.
        :param y_pred: Vector/Tensor containing predicted labels of inputs.
        :param args: Not supported arguments.
        :param kwargs: Not supported keywords arguments.
        """
        if tf.is_tensor(y_true) and tf.is_tensor(y_pred):
            y_true = y_true.numpy()
            y_pred = y_pred.numpy()

        for t_label, p_label in zip(y_true, y_pred):
            if len(self.true_labels) == self._window_size:
                old_t_label = self.true_labels.popleft()
                old_p_label = self.pred_labels.popleft()
                self._update(old_t_label, old_p_label, remove=True)
            self.true_labels.append(t_label)
            self.pred_labels.append(p_label)
            self._update(t_label, p_label)

    @abstractmethod
    def _update(self, t_label, p_label, remove: bool = False):
        """Update function that must be implemented for each metric individually,
        called during update_state(), able to update the state variables for a
        singular data point/pair, for both its addition and its removal from the window.

        :param t_label: True label of single input.
        :param p_label: Predicted label of single input.
        :param remove: Whether the input pair is to be removed from the sliding window
        or added.
        """
        raise NotImplementedError

    def merge_state(self, metrics: Self):
        """Merges the state from one or more metrics, by merging their sliding windows.

        Note this is only possible if the sliding window of the current instance is
        able to encompass all other windows.

        :param metrics: An iterable of sliding window metrics of the same type.
        """
        for m in metrics:
            self.update_state(m.true_labels, m.pred_labels)

    def reset_state(self):
        """Resets the sliding window and all the metric's state variables."""
        self.true_labels = deque()
        self.pred_labels = deque()
        self._reset()

    @abstractmethod
    def _reset(self):
        """Reset function that must be implemented for each metric individually,
        called during reset_state(), resets all underlying statistics to their
        original state.
        """
        raise NotImplementedError

    @abstractmethod
    def result(self):
        """Computes and returns the scalar value(s) of the metric. Idempotent operation
        based on the underlying state variables and the sliding window.

        :return: A scalar tensor, or a dictionary of scalar tensors.
        """
        raise NotImplementedError


class ConfMatrSlidingWindowEvaluation(SlidingWindowEvaluation):
    """Sliding window evaluation metric that computes the entire confusion matrix
    along with most(*) its metrics over the k most recent predicted binary labels to
    evaluate the model's recent performance on them in point-wise manner.
    """

    _fp: int
    _tp: int
    _fn: int
    _tn: int

    def __init__(
        self, name="conf_matrix_online_evaluation", window_size: int = None, **kwargs
    ):
        """Creates a new confusion matrix sliding window evaluation metric.

        :param name: Name of metric.
        :param window_size: Size of sliding window. If not provided, assume infinite
        window size.
        :param kwargs: Additional metric/layer keywords arguments.
        """
        super().__init__(name=name, window_size=window_size, **kwargs)

        self._fp = 0
        self._tp = 0
        self._fn = 0
        self._tn = 0

    def _update(self, t_label: bool, p_label: bool, remove: bool = False):
        """Updates the confusion matrix based on a single data point/pair, for both
        its addition and its removal from the window.

        :param t_label: True label of single input.
        :param p_label: Predicted label of single input.
        :param remove: Whether the input pair is to be removed from the sliding
        window confusion matrix or added.
        """
        mod = 1 if not remove else -1
        if t_label:
            if p_label:
                self._tp += mod
            else:
                self._fn += mod
        else:
            if not p_label:
                self._tn += mod
            else:
                self._fp += mod

    def _reset(self):
        """Zeroes the confusion matrix."""
        self._fp = 0
        self._tp = 0
        self._fn = 0
        self._tn = 0

    # noinspection DuplicatedCode
    def result(self) -> dict[str, Tensor]:
        """Based on the accumulated confusion matrix, computes its derived scalar
        metrics and returns them.

        :return: Dictionary of all derived scalar (tensor) confusion matrix metrics.
        """
        accuracy = (self._tp + self._tn) / len(self.true_labels)
        recall = tf.math.divide_no_nan(self._tp, (self._tp + self._fn))
        tnr = tf.math.divide_no_nan(self._tn, (self._tn + self._fp))
        precision = tf.math.divide_no_nan(self._tp, (self._tp + self._fp))
        npv = tf.math.divide_no_nan(self._tn, (self._tn + self._fn))
        fnr = tf.math.divide_no_nan(self._fn, (self._fn + self._tp))
        fpr = tf.math.divide_no_nan(self._fp, (self._fp + self._tn))
        f1 = tf.math.divide_no_nan(2 * self._tp, (2 * self._tp + self._fp + self._fn))

        return {
            "accuracy": accuracy,
            "recall": recall,
            "true negative rate": tnr,
            "precision": precision,
            "negative predictive value": npv,
            "false negative rate": fnr,
            "false positive rate": fpr,
            "f1": f1,
        }


class TFMetricSlidingWindowEvaluation(SlidingWindowEvaluation):
    """Wrapper class for all kinds of tensorflow evaluation metrics that operate on
    true-predicted label comparisons. Uses the provided sliding window to accumulate
    a subset of the overall data points and evaluates them using the tensorflow
    metric when called upon. Not very computational efficient since cumulative
    aggregation cannot be supported as not every metric can be computed in sliding
    window manner.
    """

    _tf_metric: keras.metrics.Metric

    def __init__(
        self, tf_metric: keras.metrics.Metric, window_size: int = None, **kwargs
    ):
        """Create a new wrapped tf sliding window evaluation metric.


        :param tf_metric: Tensorflow metric to be wrapped.
        :param window_size: Size of sliding window. If not provided, assume infinite
        window size.
        :param kwargs: Additional metric/layer keywords arguments.
        """
        super().__init__(
            name=tf_metric.name + "_online_evaluation",
            window_size=window_size,
            **kwargs,
        )

        self._tf_metric = tf_metric

    def _update(self, t_label, p_label, remove: bool = False):
        """Method is skipped as the entire metric is computed during every result()
        call."""
        pass

    def _reset(self):
        """As the metrics are reset every result() call, there is no reason to reset
        more."""
        pass

    def result(self):
        """Based on the current window, computes and returns the scalar metric value
        tensor or a dict of scalars, after resetting the metric once more.

        :return: Tensorflow metric result.
        """
        result = self._tf_metric(self.true_labels, self.pred_labels)
        self._tf_metric.reset_state()
        return result
