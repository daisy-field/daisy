# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""AutoModelScaler for choosing the most suitable local model based on the local
hardware constraints

Author: Seraphin Zunzer
Modified: 13.01.25
"""

import logging

import psutil

from daisy.personalized_fl_components.local_models import (
    TFFederatedModel_small,
    TFFederatedModel_medium,
    TFFederatedModel_large,
)


class AutoModelScaler:
    """Automatically select the most suitable federated model for a node,
    based on the locally available harware ressources"""

    _logger: logging.Logger

    def cpu_usage(self):
        """Get current CPU usage of device

        :return: CPU count, Core usages and CPU percentage
        """

        core_usages = []
        for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
            core_usages.append(percentage)
        cpu_percentage = psutil.cpu_percent()
        cpu_count = psutil.cpu_count(logical=True)
        return cpu_count, core_usages, cpu_percentage

    def ram_usage(self):
        """Get current RAM usage of device

        :return: total RAM, available RAM, used RAM, RAM percentage.
        """
        svmem = psutil.virtual_memory()
        total = svmem.total
        available = svmem.available
        used = svmem.used
        percentage = svmem.percent
        return total, available, used, percentage

    def get_manual_model(
        self, identifier, input_size, optimizer, loss, batchSize, epochs
    ):
        """
        Manually select a model size. Currently, "small", "medium" and "large" models
        are available

        :param identifier: identifier for model size, i.e. small, medium or large.
        :param input_size: input size of the model.
        :param optimizer: optimizer of the model.
        :param loss: loss function of the model.
        :param batchSize: batch size.
        :param epochs: number of epochs.
        :return: Federated Model
        """
        id_fn = None
        if identifier == "small":
            print("Creating small model")
            id_fn = TFFederatedModel_small.get_fae(
                input_size=input_size,
                optimizer=optimizer,
                loss=loss,
                batch_size=batchSize,
                epochs=epochs,
            )
        if identifier == "medium":
            print("Creating medium model")
            id_fn = TFFederatedModel_medium.get_fae(
                input_size=input_size,
                optimizer=optimizer,
                loss=loss,
                batch_size=batchSize,
                epochs=epochs,
            )
        if identifier == "large":
            print("Creating large model")
            id_fn = TFFederatedModel_large.get_fae(
                input_size=input_size,
                optimizer=optimizer,
                loss=loss,
                batch_size=batchSize,
                epochs=epochs,
            )
        return id_fn

    def choose_model(self, inputSize, optimizer, loss, batchSize, epochs):
        """Automatically chooses a Federated Model of the three available model sizes,
        by evaluating the current cpu usage.

        :param input_size: input size of the model.
        :param optimizer: optimizer of the model.
        :param loss: loss function of the model.
        :param batchSize: batch size.
        :param epochs: number of epochs.
        :return: Federated Model
        """

        _, _, _, ram_percentage = self.ram_usage()
        _, _, cpu_percentage = self.cpu_usage()
        self._logger = logging.getLogger("AutoModelScaler")

        if cpu_percentage > 90:
            self._logger.info("Auto Selected small model...")
            return TFFederatedModel_small.get_fae(
                input_size=inputSize,
                optimizer=optimizer,
                loss=loss,
                batch_size=batchSize,
                epochs=epochs,
            )
        elif cpu_percentage > 50:
            self._logger.info("Auto Selected medium model...")
            return TFFederatedModel_medium.get_fae(
                input_size=inputSize,
                optimizer=optimizer,
                loss=loss,
                batch_size=batchSize,
                epochs=epochs,
            )
        else:
            self._logger.info("Auto Selected large model...")
            return TFFederatedModel_large.get_fae(
                input_size=inputSize,
                optimizer=optimizer,
                loss=loss,
                batch_size=batchSize,
                epochs=epochs,
            )
