# Copyright (C) 2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np

from daisy.data_sources.scaler_classes.streaming_min_max_scaler import (
    StreamingMinMaxScaler,
)

# Scaler-Instanz erstellen
scaler = StreamingMinMaxScaler(feature_range=(0, 1))


def scale_data_point(o_point: object) -> object:
    """
    Skaliert die numerischen Features eines Datenpunkts (Dictionary).

    :param o_point: Datenpunkt als generisches Objekt (erwartet ein Dictionary).
    :return: Transformiertes Datenpunkt-Objekt.
    """
    # Prüfen, ob das übergebene Objekt ein Dictionary ist
    if not isinstance(o_point, dict):
        raise ValueError("scale_data_point erwartet ein Dictionary als Eingabe.")

    # Rename "old_key" to "new_key"
    if "Jammer_On" in o_point:
        o_point["label"] = o_point.pop("Jammer_On")

    # Convert all values to float
    try:
        o_point = {key: float(value) for key, value in o_point.items()}
    except Exception as e:
        print(f"Error: {e}")

    # Extrahiere die numerischen Features
    # keys = [key for key in o_point if key != "label"]
    features = np.array([o_point[key] for key in o_point if key != "label"])

    # Fitte und skaliere die Features
    scaler.partial_fit([features])  # Update Min/Max-Werte
    scaled_features = scaler.transform([features])[0]

    # Aktualisiere den Datenpunkt mit den skalierten Werten
    for i, key in enumerate([k for k in o_point if k != "label"]):
        o_point[key] = scaled_features[i]

    return o_point
