# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Model poisoning scenario, where different untargeted model poisoning attacks can be
executed to harm the FL process. Depending on the specified poisoning mode, node where
model poisoning is enabled either randomizes, zeros or inverts the calculated local
model weights. When inserted into the FL process, these manipulated models should
decrease the overall global model performance.

Author: Seraphin Zunzer
Modified: 02.03.25
"""

import logging

import numpy as np
from enum import Enum


class PoisoningMode(Enum):
    NONE = ("No model poisoning", False, True)
    ZERO = ("All zeros poisoning", True, False)
    INVERSE = ("Invert local model", False, True)
    RANDOM = ("Randomizes local model", True, False)

    def __init__(
        self, description: str, overwrite_local_model: bool, accept_global_model: bool
    ):
        self.description = description
        self.overwrite_local_model = overwrite_local_model
        self.accept_global_model = accept_global_model


class ModelPoisoning:
    poisoning_mode: PoisoningMode

    def __init__(
        self,
        poisoning_mode: PoisoningMode,
        name: str = "ModelPoisoning",
        log_level: int = None,
    ):
        self._logger = logging.getLogger(name)
        if log_level:
            self._logger.setLevel(log_level)

        self.poisoning_mode = poisoning_mode

    def get_poisoned_parameters(self, parameters: list):
        match self.poisoning_mode:
            case PoisoningMode.NONE:
                return parameters
            case PoisoningMode.ZERO:
                return self._zero_poisoning(parameters)
            case PoisoningMode.INVERSE:
                return self._inverse_poisoning(parameters)
            case PoisoningMode.RANDOM:
                return self._random_poisoning(parameters)
            case _:
                raise NotImplementedError("Unknown poisoning mode")

    @staticmethod
    def _zero_poisoning(parameters: list):
        poisoned_parameters = []
        for layer in parameters:
            poisoned_parameters.append(
                0 if isinstance(layer, (int, float)) else np.zeros_like(layer)
            )
        return poisoned_parameters

    @staticmethod
    def _inverse_poisoning(parameters: list):
        poisoned_parameters = []
        for layer in parameters:
            poisoned_parameters.append(layer * -1)
        return poisoned_parameters

    @staticmethod
    def _random_poisoning(parameters: list):
        poisoned_parameters = []
        rand = np.random
        for layer in parameters:
            poisoned_parameters.append(
                rand.random()
                if isinstance(layer, (int, float))
                else rand.random_sample(layer.shape)
            )
        return poisoned_parameters
