"""
    Class for MetricsObject, an object providing access to various metrics
    for a given prediction and corresponding true labels list.

    Author: Seraphin Zunzer
    Modified: 09.08.23
"""

import logging


class MetricsObject():
    """ Object of evaluation metrics"""

    metrics = []

    def __init__(self, prediction: [], true_labels: [], anomaly: str, normal: str):
        """Calculate confusion matrix

        :return: False positives, True positives, False negatives, True negatives
        """

        logging.info("Evaluation Object created")
        self.prediction = prediction
        self.true_labels = true_labels
        self.normal = normal
        self.anomaly = anomaly
        fp = 0
        tp = 0
        fn = 0
        tn = 0
        for i in range(0, len(self.prediction)):
            if self.prediction[i] == self.anomaly and self.true_labels[i] == self.normal:
                fp += 1
            elif self.prediction[i] == self.anomaly and self.true_labels[i] == self.anomaly:
                tp += 1
            elif self.prediction[i] == self.normal and self.true_labels[i] == self.anomaly:
                fn += 1
            elif self.prediction[i] == self.normal and self.true_labels[i] == self.normal:
                tn += 1
        self.metrics = [fp, tp, fn, tn]

    def get_confusion_matrix(self):
        """Getter for confusion matrix

        :return: List: [False Positives, True Positives, False Negatives, True Negatives]
        """
        return self.metrics

    def false_positive_r(self):
        """ Calculate False Positive Rate

        :return: False Positive Rate
        """
        fp = self.metrics[0]
        tn = self.metrics[3]
        return (fp + tn) and fp / (fp + tn)

    def true_positive_r(self):
        """ Calculate True Positive Rate

        :return: True Positive Rate
        """
        tp = self.metrics[1]
        fn = self.metrics[2]
        return (tp + fn) and tp / (tp + fn)

    def accuracy(self):
        """ Calculate Accuracy

        :return: Accuracy
        """
        fp = self.metrics[0]
        tp = self.metrics[1]
        fn = self.metrics[2]
        tn = self.metrics[3]
        return (tp + tn) / (tp + tn + fp + fn)

    def f1(self):
        fp = self.metrics[0]
        tp = self.metrics[1]
        fn = self.metrics[2]
        try:
            return (2 * tp) / ((2 * tp) + fp + fn)
        except ZeroDivisionError:
            return 0



def overall_performance(self, eval_obj_list: []):
    """Calculate the average performance of multiple metrics objects

    :return:
    """
    all_metrics = [0, 0, 0, 0]
    for i in eval_obj_list:
        [sum(x) for x in zip(all_metrics, i.metrics)]

    overall_fp = all_metrics[0]
    overall_tp = all_metrics[1]
    overall_fn = all_metrics[2]
    overall_tn = all_metrics[3]

    overall_fpr = (overall_fp + overall_tn) and overall_fp / (overall_fp + overall_tn)
    overall_tpr = (overall_tp + overall_fn) and overall_tp / (overall_tp + overall_fn)
    overall_fnr = (overall_fn + overall_tp) and overall_fn / (overall_fn + overall_tp)
    overall_ac = (overall_tp + overall_tn + overall_fp + overall_fn) and (overall_tp + overall_tn) / (
            overall_tp + overall_tn + overall_fp + overall_fn)

    return overall_fpr, overall_tpr, overall_fnr, overall_ac