"""
    Functions to calculate and analyze MAD score

    Author: Seraphin Zunzer
    Modified: 09.08.23
"""
import numpy as np

from utils.metrics import MetricsObject


def calculate_mad_score(points:[]):
    """
    Calculate MAD Score for given list of points

    :param points: List of points
    :return: List of MAD scores
    """
    m = np.median(points)
    ad = np.abs(points - m)
    mad = np.median(ad)
    return 0.6745 * ad / mad




def analyze_mad_score(z_scores:[], labels_true:[]):
    """Analyze MAD threshold. Set different thresholds and calculate True positive rate.
    Based on the convergence of th TPR, you can set the MAD threshold accordingly.

    :param z_scores: Z_scores list
    :param labels_true: True labels list
    :return: Write TPR for different MAD threshold to file
    """
    tpr_i = []
    i = 0
    while i < 10:
        outliers = z_scores > i
        pred = ["BENIGN" if not l else "ANOMALY" for l in outliers]
        fp, tp, fn, tn = MetricsObject(pred, labels_true, "Normal", "Anomaly").get_confusion_matrix()
        fpr = fp / (fp + tn)
        tpr_i.append(fpr)
        i += 0.1
    with open('results/mad.txt', 'w') as file:
        for i, tpr in enumerate(tpr_i):
            file.write(f'{round(i * 0.1, 2)}	{tpr}\n')
