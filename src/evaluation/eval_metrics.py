"""
    TODO

        Class for MetricsObject, an object providing access to various metrics
    for a given prediction and corresponding true labels list.

    Author: Fabian Hofmann, Seraphin Zunzer
    Modified: 15.08.23

"""
import queue
from abc import ABC, abstractmethod
from typing import Optional, Self

from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras


class ADOnlineEvaluation(keras.metrics.Metric, ABC):
    """TODO

    """
    true_labels: deque
    pred_labels: deque
    _window_size: int

    def __init__(self, name='ad_online_evaluation', window_size: int = 0, **kwargs):
        """TODO

        :param name:
        :param window_size:
        :param kwargs:
        """
        super().__init__(name=name, **kwargs)

        self.true_labels = deque()
        self.pred_labels = deque()
        self._window_size = window_size

    def update_state(self, y_true, y_pred, *args, **kwargs):
        """TODO

        :param y_true:
        :param y_pred:
        :param args:
        :param kwargs:
        :return:
        """
        for t_label, p_label in zip(y_true, y_pred):
            if len(self.true_labels) == self._window_size:
                old_t_label = self.true_labels.popleft()
                old_p_label = self.pred_labels.popleft()
                self._update(old_t_label, old_p_label, remove=True)
            self.true_labels.append(t_label)
            self.pred_labels.append(p_label)
            self._update(t_label, p_label, remove=True)

    @abstractmethod
    def _update(self, t_label, p_label, remove: bool = False):
        """TODO

        :param t_label:
        :param p_label:
        :param remove:
        """
        raise NotImplementedError

    def merge_state(self, metrics: Self):
        """TODO

        :param metrics:
        :return:
        """
        for m in metrics:
            self.update_state(m.true_labels, m.pred_labels)

    def reset_state(self):
        """TODO

        :return:
        """
        self.true_labels = deque()
        self.pred_labels = deque()

    @abstractmethod
    def result(self):
        """TODO

        :return:
        """
        raise NotImplementedError


class ConfMatrixOnlineEvaluation(ADOnlineEvaluation):
    """Helper class to evaluate anomaly detection results, by comparing the predictions with the true labels. Computes
    helpful evaluation metrics in a stateless manner.

    TODO
    """
    _fp: int
    _tp: int
    _fn: int
    _tn: int

    def __init__(self, name='conf_matrix_online_evaluation', window_size: int = 0, **kwargs):
        """TODO Create a new anomaly detection evaluator that is loaded with the true labels of the dataset.

        :param window_size:
        """
        #         """Calculate confusion matrix
        #
        # :return: False positives, True positives, False negatives, True negatives
        # """
        super().__init__(name=name, window_size=window_size, **kwargs)

        self._fp = 0
        self._tp = 0
        self._fn = 0
        self._tn = 0

    def update_state(self, y_true, y_pred, *args, **kwargs):
        """

        :param y_true:
        :param y_pred:
        :param args:
        :param kwargs:
        :return:
        """
        y_true = tf.cast(y_true, tf.bool).numpy()
        y_pred = tf.cast(y_pred, tf.bool).numpy()
        super().update_state(y_true, y_pred, *args, **kwargs)

    def _update(self, t_label: bool, p_label: bool, remove: bool = False):
        """TODO

        :param t_label:
        :param p_label:
        :param remove:
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

    def reset_state(self):
        """TODO

        :return:
        """
        self._fp = 0
        self._tp = 0
        self._fn = 0
        self._tn = 0

    def result(self) -> dict[str, float]:
        """TODO

        :return:
        """
        accuracy = (self.tp + self.tn) / len(self.true_labels)
        recall = self.tp / (self.tp + self.fn)
        tnr = self.tn / (self.tn + self.fp)
        precision = self.tp / (self.tp + self.fp)
        npv = self.tn / (self.tn + self.fn)
        fnr = self.fn / (self.fn + self.tp)
        fpr = self.fp / (self.fp + self.tn)
        f1 = 2 * self.tp / (2 * self.tp + self.fp + self.fn)

        return {"accuracy": accuracy, "recall": recall, "true negative rate": tnr,
                "precision": precision, "negative predictive value": npv,
                "false negative rate": fnr, "false positive rate": fpr, "f1 measure": f1}


class TFMetricOnlineEvaluation(ADOnlineEvaluation):
    """TODO

    """
    _tf_metric: keras.metrics.Metric

    def __init__(self, tf_metric: keras.metrics.Metric, window_size: int = 0, **kwargs):
        """TODO Create a new anomaly detection evaluator that is loaded with the true labels of the dataset.

        :param window_size:
        """
        #         """Calculate confusion matrix
        #
        # :return: False positives, True positives, False negatives, True negatives
        # """
        super().__init__(name=tf_metric.name + '_online_evaluation', window_size=window_size, **kwargs)

        self._tf_metric = tf_metric

    def _update(self, t_label, p_label, remove: bool = False):
        """TODO

        :param t_label:
        :param p_label:
        :param remove:
        :return:
        """
        pass

    def result(self):
        """TODO

        :return:
        """
        result = self._tf_metric.__call__(self.true_labels, self.pred_labels)
        self._tf_metric.reset_state()
        return result
