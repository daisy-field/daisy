"""
    TODO

        Class for MetricsObject, an object providing access to various metrics
    for a given prediction and corresponding true labels list.

    Author: Fabian Hofmann, Seraphin Zunzer
    Modified: 15.08.23
"""
import queue

from typing import Optional, Self


class AnomalyDetectionEvaluation:
    """Helper class to evaluate anomaly detection results, by comparing the predictions with the true labels. Computes
    helpful evaluation metrics in a stateless manner.

    TODO
    """
    _predictions: queue.Queue[tuple[bool, bool]]

    _n: int
    _p: int
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
        self._predictions = queue.Queue(window_size)

        self._n = 0
        self._p = 0
        self._fp = 0
        self._tp = 0
        self._fn = 0
        self._tn = 0

    def get_eval(self, *predictions: tuple[bool, bool], online: bool = False) -> Optional[dict[str, float]]:
        """TODO

        :param predictions:
        :param online:
        :return:
        """

        #     """Computes several evaluation metrics by comparing the predicted to the actual labels.
        #
        #     :param predicted_labels: Array with predicted labels (boolean).
        #     :return:
        #         - accuracy
        #         - recall (true positive rate)
        #         - true negative rate
        #         - precision (positive predictive value)
        #         - negative predictive value
        #         - false negative rate
        #         - false positive rate
        #         - f1 measure
        #     """
        for new_prediction in predictions:
            if self._predictions.full():
                self._update(*self._predictions.get(), remove=True)
            self._predictions.put(new_prediction)
            self._update(*new_prediction)

        if online:
            accuracy = (self._tp + self._tn) / self._predictions.qsize()
            recall = self._tp / (self._tp + self._fn)
            tnr = self._tn / (self._tn + self._fp)
            precision = self._tp / (self._tp + self._fp)
            npv = self._tn / (self._tn + self._fn)
            fnr = self._fn / (self._fn + self._tp)
            fpr = self._fp / (self._fp + self._tn)
            f1 = 2 * self._tp / (2 * self._tp + self._fp + self._fn)

            return {"accuracy": accuracy, "recall": recall, "true negative rate": tnr,
                    "precision": precision, "negative predictive value": npv,
                    "false negative rate": fnr, "false positive rate": fpr, "f1 measure": f1}

    def _update(self, t_label: bool, p_label: bool, remove: bool = False):
        """TODO

        :param t_label:
        :param p_label:
        :param remove:
        """
        mod = 1 if not remove else -1
        if t_label:
            self._p += mod
            if p_label:
                self._tp += mod
            else:
                self._fn += mod
        else:
            self._n += mod
            if not p_label:
                self._tn += mod
            else:
                self._fp += mod

    @classmethod
    def merge_eval_results(cls, *eval_results: Self) -> dict[str, float]:
        """TODO

        :param eval_results:
        :return:
        """
        merged_eval = AnomalyDetectionEvaluation()
        for eval_result in eval_results:
            merged_eval._n += eval_result._n
            merged_eval._p += eval_result._p
            merged_eval._fp += eval_result._fp
            merged_eval._tp += eval_result._tp
            merged_eval._fn += eval_result._fn
            merged_eval._tn += eval_result._tn
        return merged_eval.get_eval()
