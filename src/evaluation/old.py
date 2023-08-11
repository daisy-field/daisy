from typing import Mapping

import numpy as np


class AnomalyDetectionEvaluation:
    """Helper class to evaluate anomaly detection results, by comparing the predictions with the true labels. Computes
    helpful evaluation metrics in a stateless manner.
    """
    true_labels: np.ndarray
    total: int
    n: int
    p: int

    def __init__(self, true_labels_path: str):
        """Create a new anomaly detection evaluator that is loaded with the true labels of the dataset.

        :param true_labels_path: Path to csv with actual labels (boolean).
        """
        self.true_labels = np.genfromtxt(true_labels_path, delimiter=";", converters={0: lambda b: bool(int(b))})
        self.total = len(self.true_labels)
        self.n, self.p = tuple(np.bincount(self.true_labels))

    def evaluate(self, predicted_labels: np.ndarray) -> Mapping[str, float]:
        """Computes several evaluation metrics by comparing the predicted to the actual labels.

        :param predicted_labels: Array with predicted labels (boolean).
        :return:
            - accuracy
            - recall (true positive rate)
            - true negative rate
            - precision (positive predictive value)
            - negative predictive value
            - false negative rate
            - false positive rate
            - f1 measure
        """
        if self.total != len(predicted_labels):
            raise ValueError("Dimensions of true and predicted labels do not match: {}, {}"
                             .format(self.total, len(predicted_labels)))

        pn, pp = tuple(np.bincount(predicted_labels))

        tp = np.bincount(np.logical_and(self.true_labels, predicted_labels))[1]
        tn = np.bincount(np.logical_and(np.logical_not(self.true_labels), np.logical_not(predicted_labels)))[1]
        fp = pp - tp
        fn = pn - tn

        accuracy = (tp + tn) / self.total
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