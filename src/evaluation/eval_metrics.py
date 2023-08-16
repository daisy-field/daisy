"""
    TODO

        Class for MetricsObject, an object providing access to various metrics
    for a given prediction and corresponding true labels list.

    Author: Fabian Hofmann, Seraphin Zunzer
    Modified: 15.08.23

    TODO Future Work should be the implementation of Open Source Interfaces (e.g. Keras Metric API)
"""
import queue
from abc import ABC, abstractmethod
from typing import Optional, Self

import numpy as np
from tensorflow import keras


# TODO queue replaced with lists?

class ADOnlineEvaluation(ABC):
    """TODO

    """
    _predictions: queue.Queue[tuple[float, float]]

    def __init__(self, window_size: int = 0):
        """TODO

        :param window_size:
        """
        self._predictions = queue.Queue(window_size)

    def __call__(self, *predictions: tuple[float, float], get_eval: bool = False) -> Optional[dict[str, float]]:
        """TODO

        :param predictions:
        :param get_eval:
        :return:
        """
        for new_prediction in predictions:
            if self._predictions.full():
                self._update(*self._predictions.get(), remove=True)
            self._predictions.put(new_prediction)
            self._update(*new_prediction)

        if get_eval:
            return self.get_eval()

    @abstractmethod
    def _update(self, t_label: float, p_label: float, remove: bool = False):
        """TODO

        :param t_label:
        :param p_label:
        :param remove:
        """
        raise NotImplementedError

    @abstractmethod
    def get_eval(self) -> dict[str, float]:
        """TODO

        :return:
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def merge_eval_results(cls, *eval_results: Self) -> dict[str, float]:
        """TODO

        :param eval_results:
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

    def __init__(self, window_size: int = 0):
        """TODO Create a new anomaly detection evaluator that is loaded with the true labels of the dataset.

        :param window_size:
        """
        #         """Calculate confusion matrix
        #
        # :return: False positives, True positives, False negatives, True negatives
        # """
        super().__init__(window_size)

        self._fp = 0
        self._tp = 0
        self._fn = 0
        self._tn = 0

    def _update(self, t_label: float, p_label: float, remove: bool = False):
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

    def get_eval(self) -> dict[str, float]:
        """TODO

        :return:
        """
        return compute_conf_metrics(self._predictions.qsize(), self._tp, self._tn, self._fp, self._fn)

    @classmethod
    def merge_eval_results(cls, *eval_results: Self) -> dict[str, float]:
        """TODO

        :param eval_results:
        :return:
        """
        count = 0
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for eval_result in eval_results:
            count += eval_result._predictions.qsize()
            tp += eval_result._tp
            tn += eval_result._tn
            fp += eval_result._fp
            fn += eval_result._fn
        return compute_conf_metrics(count, tp, tn, fp, fn)


def compute_conf_metrics(count, tp, tn, fp, fn) -> dict[str, float]:
    """TODO

    :param count:
    :param tp:
    :param tn:
    :param fp:
    :param fn:
    :return:
    """
    accuracy = (tp + tn) / count
    recall = tp / (tp + fn)
    tnr = tn / (tn + fp)
    precision = tp / (tp + fp)
    npv = tn / (tn + fn)
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)
    f1 = 2 * tp / (2 * tp + fp + fn)

    return {"accuracy": accuracy, "recall": recall, "true negative rate": tnr,
            "precision": precision, "negative predictive value": npv,
            "false negative rate": fnr, "false positive rate": fpr, "f1 measure": f1}


class AUCOnlineEvaluation(ADOnlineEvaluation):
    """TODO

    """
    _num_thresholds: int

    def __init__(self, num_thresholds: int = 10, window_size: int = 0):
        """TODO Create a new anomaly detection evaluator that is loaded with the true labels of the dataset.

        :param window_size:
        """
        #         """Calculate confusion matrix
        #
        # :return: False positives, True positives, False negatives, True negatives
        # """
        super().__init__(window_size)

        self._num_thresholds = num_thresholds

    def _update(self, t_label: float, p_label: float, remove: bool = False):
        """TODO

        :param t_label:
        :param p_label:
        :param remove:
        :return:
        """
        pass

    def get_eval(self) -> dict[str, float]:
        """

        """
        predictions = []
        while not self._predictions.empty():
            predictions.append(self._predictions.get())
        predictions = np.array(predictions).T
        y_true, y_pred = predictions[0], predictions[0]
        return {"area under curve": keras.metrics.AUC(num_thresholds=self._num_thresholds)(y_true, y_pred)}

    @classmethod
    def merge_eval_results(cls, *eval_results: Self, num_thresholds: int = 10) -> dict[str, float]:
        """

        """
        predictions = []
        for eval_result in eval_results:
            while not eval_result.empty():
                predictions.append(eval_result.get())
        predictions = np.array(predictions).T
        y_true, y_pred = predictions[0], predictions[1]
        return {"area under curve": keras.metrics.AUC(num_thresholds=num_thresholds)(y_true, y_pred)}
