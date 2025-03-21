# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Model poisoning scenario, where different untargeted model poisoning attacks can be executed to harm the FL process.
Depending on the specified poisoning mode, node where model poisoning is enabled either randomizes, zeros or inverts
the calculated local model weights. When inserted into the FL process, these manipulated models should decrease the
overall global model performance

Author: Seraphin Zunzer
Modified: 02.03.25
"""

import numpy as np


def model_poisoning(self, current_params, poisoning_mode, model):  # TODO self???
    """Function for poisoning model weights. Based on the current parameters and the poisoning mode, the
    function either returns a list of random, zeroed or inverted parameters with the same shape
    as the original parameters. If poisoning mode is none, the original parameters are returned.
    Additionally, the model weights of the nodes local model are modified to the manipulated weights,
    to be able to visualize the decreasing performance of the manipulated node. This is currently only done
    for random and zero poisoning.

    :param model: Nodes local detection model, weights are set using the set_weights() function
    :param current_params: List of current model weights.
    :param poisoning_mode: Poisoning mode, either None, zeros, random or inverse.
    :return: poisoned parameters
    """

    poisoned_params = []

    if poisoning_mode is None:
        self._logger.info("Send unpoisoned parameters")
        return current_params
    else:
        if poisoning_mode == "zeros":
            for layer in current_params:
                if isinstance(layer, int) or isinstance(layer, float):
                    poisoned_params.append(0)
                else:
                    poisoned_params.append(np.zeros_like(layer))
            model.set_parameters(poisoned_params)
            self._logger.info("Model poisoning: Set weights to zero")

        elif poisoning_mode == "random":
            for layer in current_params:
                if isinstance(layer, int) or isinstance(layer, float):
                    poisoned_params.append(np.random.random())
                else:
                    poisoned_params.append(np.random.random_sample(layer.shape))
            model.set_parameters(poisoned_params)
            self._logger.info("Model poisoning: Set weights random")

        elif self._poisoningMode == "inverse":
            for layer in current_params:
                poisoned_params.append(layer * -1)
            # Skipped setting new parameters, as we need to inverse the real parameters next round

        self._logger.info("Poisoned Parameters:")
        for i in poisoned_params:
            if not (isinstance(i, int) or isinstance(i, float)):
                self._logger.info(i.shape)
            else:
                self._logger.info(i)
        self._logger.info("Send poisoned params to model aggregation server...")
        return poisoned_params
