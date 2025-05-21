# Copyright (C) 2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from typing import Tuple, Union, List
import numpy as np


class StreamingMinMaxScaler:
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        # preload with knowledge from HMF data
        self.min_val = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.8]
        self.max_val = [15, 2, 28, 3192, 4291, 30340, 491, 57, 25160, 1803227, 22204, 4736, 0, 100, 39.6]
        self.feature_range = feature_range
        self.range_diff = abs(feature_range[1] - feature_range[0])

    def partial_fit(self, X: Union[List[float], np.ndarray]):
        """
        Aktualisiert die Min- und Max-Werte basierend auf den übergebenen Daten.

        :param X: Einzelner Datenpunkt oder Batch von Datenpunkten.
        """
        X = np.array(X)  # Sicherstellen, dass es ein Array ist
        if self.min_val is None or self.max_val is None:
            self.min_val = X.min(axis=0)
            self.max_val = X.max(axis=0)
        else:
            self.min_val = np.minimum(self.min_val, X.min(axis=0))
            self.max_val = np.maximum(self.max_val, X.max(axis=0))

    def transform(self, X: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Skaliert die Daten basierend auf den aktuellen Min- und Max-Werten.

        :param X: Einzelner Datenpunkt oder Batch von Datenpunkten.
        :return: Skalierte Daten als Array.
        """
        X = np.array(X)  # Sicherstellen, dass es ein Array ist
        # Vermeiden von Division durch null
        range_diff = self.max_val - self.min_val
        range_diff[range_diff == 0] = 1  # Setze 1 für identische Min- und Max-Werte
        scaled = (X - self.min_val) / range_diff
        scaled = scaled * self.range_diff + self.feature_range[0]
        return scaled

    def fit_transform(self, X: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Führt `partial_fit` und `transform` nacheinander aus.

        :param X: Einzelner Datenpunkt oder Batch von Datenpunkten.
        :return: Skalierte Daten als Array.
        """
        self.partial_fit(X)
        return self.transform(X)
