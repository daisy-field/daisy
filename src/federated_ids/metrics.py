




class MetricsObject():
    """ Object latest evaluation metrics
    """
    metrics = []

    def __init__(self, prediction: [], true_labels: [], anomaly: str, normal: str):
        """Calculate confusion matrice

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

    def f1(selfself):
        fp = self.metrics[0]
        tp = self.metrics[1]
        fn = self.metrics[2]
        try:
            return (2 * tp) / ((2 * tp) + fp + fn)
        except ZeroDivisionError:
            return  0
