
import numpy as np

def model_poisoning(self, current_params, poisoning_mode, model):
    """
    Function for poisoning model weights. Based on the current params and the poisoning mode, the
    function either retruns a list of random values, zeros or inverted parameters with the same shape
    as the original parameters. If poisoning mode is none, the original parameters are returned.

    :param model:
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
            model.set_parameters(poisoned_params)  # new_params)
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

