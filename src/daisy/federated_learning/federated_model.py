# Copyright (C) 2024-2025 DAI-Labor and others
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""A collection of various types of model wrappers, implementing the same interface
for each federated model type, thus enabling their inter-compatibility for different
aggregation strategies and federated system components.

Author: Fabian Hofmann
Modified: 04.04.24
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Self

import keras
import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from daisy.federated_learning.model_classes.vae import DetectorVAE

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix


class FederatedModel(ABC):
    """An abstract model wrapper that offers the same methods, no matter the type of
    underlying model. Must always be implemented if a new model type is to be used in
    the federated system.
    """

    @abstractmethod
    def set_parameters(self, parameters: list[np.ndarray]):
        """Updates the internal parameters of the model.

        :param parameters: Parameters to update the model with.
        """
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self) -> list[np.ndarray]:
        """Retrieves the internal parameters of the model.

        :return: Parameters of model.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, x_data, y_data):
        """Trains the model with the given data, which must be compatible with the
        tensorflow API
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).

        :param x_data: Input data.
        :param y_data: Expected output.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_data) -> Tensor:
        """Makes a prediction on the given data and returns it, which must be
        compatible with the tensorflow API
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict).

        :param x_data: Input data.
        :return: Predicted output tensor.
        """
        raise NotImplementedError


class TFFederatedModel(FederatedModel):
    """The standard federated model wrapper for tensorflow models. Can be used for
    both online and offline training/ prediction, by default online, however.
    """

    _model: keras.Model
    _batch_size: int
    _epochs: int

    def __init__(
        self,
        model: keras.Model,
        optimizer: str | keras.optimizers.Optimizer,
        loss: str | keras.losses.Loss,
        metrics: list[str | Callable | keras.metrics.Metric] = None,
        batch_size: int = 32,
        epochs: int = 1,
    ):
        """Creates a new tensorflow federated model from a given model. This also
        compiles the given model, requiring a set of additional arguments
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile).

        :param model: Underlying model to be wrapped around.
        :param optimizer: Optimizer to use during training.
        :param loss: Loss function to use during training.
        :param metrics: Evaluation metrics to be displayed during training and testing.
        :param batch_size: Batch size during training and prediction.
        :param epochs: Number of epochs (rounds) during training.
        """
        self._model = model
        # behandelt dynamische custom loss Funktionen anderes als statische standard loss Funktionen
        if (
            isinstance(
                loss, keras.losses.Loss
            )  # Klassenbasierte benutzerdefinierte Verluste
            or callable(loss)  # Callable Funktionen (inkl. Lambda)
            or isinstance(loss, tf.Tensor)  # TensorFlow Tensoren
            or tf.is_tensor(loss)  # Alle TensorFlow-Tensoren, inkl. KerasTensor
            or isinstance(
                loss, keras.layers.Layer
            )  # Benutzerdefinierte Schichten, die `add_loss` verwenden
            or hasattr(loss, "__call__")
        ):  # Objekte mit einer __call__-Methode
            self._model.add_loss(loss)
            self._model.compile(optimizer=optimizer, metrics=metrics or [])
        else:
            self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics or [])
        self._batch_size = batch_size
        self._epochs = epochs

    def set_parameters(self, parameters: list[np.ndarray]):
        """Updates the weights of the underlying model with new ones.

        :param parameters: Weights to update the model with.
        """
        self._model.set_weights(parameters)

    def get_parameters(self) -> list[np.ndarray]:
        """Retrieves the weights of the underlying model.

        :return: Weights of the model.
        """
        return self._model.get_weights()

    def fit(self, x_data, y_data):
        """Trains the model with the given data by calling the wrapped model,
        which must be compatible with the tensorflow API
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).

        :param x_data: Input data.
        :param y_data: Expected output.
        """
        history = self._model.fit(
            x=x_data,
            y=y_data,
            batch_size=self._batch_size,
            epochs=self._epochs,
            validation_split=0.1,
            verbose=1,
        )
        return history

    def predict(self, x_data) -> Tensor:
        """Makes a prediction on the given data and returns it by calling the wrapped
        model. Uses the call() tensorflow model interface for small numbers of data
        points, which must be compatible with the tensorflow API (see:
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict).

        :param x_data: Input data.
        :return: Predicted output tensor.
        """
        if (
            len(x_data) >= self._batch_size
        ):  # was ist der unterschied dieser zwei aufrufe? müsste da nicht >= sein. Es war =
            return self._model.predict(x=x_data, batch_size=self._batch_size)
        return self._model(x_data, training=False).numpy()

    @classmethod
    def get_fae(
        cls,
        input_size: int,
        optimizer: str | keras.optimizers.Optimizer = "Adam",
        loss: str | keras.losses.Loss = "mse",
        metrics: list[str | Callable | keras.metrics.Metric] = None,
        batch_size: int = 32,
        epochs: int = 1,
    ) -> Self:
        """Factory class method to create a simple federated autoencoder model of a
        fixed depth but with variable input size.

        Should only serve as a quick and basic setup for a model.

        :param input_size: Dimensionality of input/output of autoencoder.
        :param optimizer: Optimizer to use during training.
        :param loss: Loss function to use during training.
        :param metrics: Evaluation metrics to be displayed during training and testing.
        :param batch_size: Batch size during training and prediction.
        :param epochs: Number of epochs (rounds) during training.
        :return: Initialized federated autoencoder model.
        """
        enc_inputs = keras.layers.Input(shape=(input_size,))
        x = keras.layers.Dense(input_size)(enc_inputs)
        x = keras.layers.Dense(35)(x)
        enc_outputs = keras.layers.Dense(18)(x)
        encoder = keras.Model(inputs=enc_inputs, outputs=enc_outputs)

        dec_inputs = keras.layers.Input(shape=(18,))
        y = keras.layers.Dense(35)(dec_inputs)
        y = keras.layers.Dense(input_size)(y)
        dec_outputs = keras.layers.Activation("sigmoid")(y)
        decoder = keras.Model(inputs=dec_inputs, outputs=dec_outputs)

        fae_inputs = keras.Input(shape=(input_size,))
        encoded = encoder(fae_inputs)
        fae_outputs = decoder(encoded)
        fae = keras.models.Model(inputs=fae_inputs, outputs=fae_outputs)
        return TFFederatedModel(fae, optimizer, loss, metrics, batch_size, epochs)

    @classmethod
    def get_fvae(
        cls,
        input_size: int,
        latent_dim: int = 4,
        hidden_layers: list[int] = [15, 12],
        optimizer: str | keras.optimizers.Optimizer = "Adam",
        metrics: list[str | Callable | keras.metrics.Metric] = None,
        batch_size: int = 32,
        epochs: int = 1,
        load_pretrained_path=None,
    ) -> Self:
        """
        Factory class method to create a simple Variational Autoencoder (VAE) model
        with a fixed architecture and variable input size.

        :param input_size: Dimensionality of the input/output of the VAE.
        :param latent_dim: Dimensionality of the latent space.
        :param optimizer: Optimizer to use during training.
        :param loss: Loss function to use during training.
        :param metrics: Evaluation metrics to be displayed during training and testing.
        :param batch_size: Batch size during training and prediction.
        :param epochs: Number of epochs during training.
        :return: Initialized VAE model.
        """

        if load_pretrained_path is None:
            vae_detector = DetectorVAE(
                input_size, hidden_layers=hidden_layers, latent_dim=latent_dim
            )
        else:
            vae_detector = DetectorVAE.load(load_pretrained_path)

        vae = vae_detector.model
        loss = vae_detector.loss

        vae.summary()
        vae_detector.encoder.summary()
        vae_detector.decoder.summary()

        return TFFederatedModel(vae, optimizer, loss, metrics, batch_size, epochs)

    @classmethod
    def get_ftae(
        cls,
        input_size: int,
        optimizer: str | keras.optimizers.Optimizer = "Adam",
        loss: str | keras.losses.Loss = "mse",
        metrics: list[str | Callable | keras.metrics.Metric] = None,
        batch_size: int = 32,
        epochs: int = 1,
    ) -> Self:
        """Factory class method to create a simple federated transformer autoencoder model of a
        fixed depth but with variable input size.

        :param input_size: Dimensionality of input/output of autoencoder.
        :param optimizer: Optimizer to use during training.
        :param loss: Loss function to use during training.
        :param metrics: Evaluation metrics to be displayed during training and testing.
        :param batch_size: Batch size during training and prediction.
        :param epochs: Number of epochs (rounds) during training.
        :return: Initialized federated autoencoder model.
        """

        enc_inputs = keras.layers.Input(
            shape=(input_size,)
        )  # input_dim ist die Anzahl der Merkmale
        x = keras.layers.Dense(32)(enc_inputs)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

        # Expandieren der Dimensionen, damit die Eingabe kompatibel ist mit MultiHeadAttention
        x = tf.expand_dims(x, axis=1)  # Form: (batch_size, 1, feature_dim)

        for _ in range(4):
            attn_output = keras.layers.MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
            x = keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Entfernen der zusätzlichen Dimension, bevor die Dense-Schicht verarbeitet wird
        x = tf.squeeze(x, axis=1)  # Form: (batch_size, feature_dim)

        # Ausgabeschicht
        outputs = keras.layers.Dense(units=input_size)(x)  # Rekonstruktion der Eingabe

        ftae = keras.models.Model(enc_inputs, outputs)

        ftae.summary()

        return TFFederatedModel(ftae, optimizer, loss, metrics, batch_size, epochs)

    @classmethod
    def get_fmlp(
        cls,
        input_size: int,
        dense_layers: list[int] = [15, 12],
        optimizer: str | keras.optimizers.Optimizer = "Adam",
        loss: str | keras.losses.Loss = "mse",
        metrics: list[str | Callable | keras.metrics.Metric] = None,
        batch_size: int = 32,
        epochs: int = 1,
    ) -> Self:
        """Factory class method to create a simple federated autoencoder model of a
        fixed depth but with variable input size.

        Should only serve as a quick and basic setup for a model.

        :param input_size: Dimensionality of input/output of autoencoder.
        :param optimizer: Optimizer to use during training.
        :param loss: Loss function to use during training.
        :param metrics: Evaluation metrics to be displayed during training and testing.
        :param batch_size: Batch size during training and prediction.
        :param epochs: Number of epochs (rounds) during training.
        :return: Initialized federated autoencoder model.
        """
        model = keras.models.Sequential()
        model.add(
            keras.layers.Dense(
                dense_layers[0], input_shape=(input_size,), activation="relu"
            )
        )
        for units in dense_layers[1:]:
            model.add(keras.layers.Dense(units, activation="relu"))
            # model.add(Dropout(0.2))
        model.add(keras.layers.Dense(1, activation="sigmoid"))
        return TFFederatedModel(model, optimizer, loss, metrics, batch_size, epochs)


class FederatedIFTM(FederatedModel):
    """Double union of two federated models, following the IFTM hybrid  model
    approach --- identify function threshold model principle by Schmidt et al.
    (https://ieeexplore.ieee.org/document/8456348): One for the computation of the
    identity of a given sample (alternatively prediction of the next sample),
    while the other maps the error/loss using a threshold(-model) to the binary class
    labels for anomaly detection. Both are generic federated models, so any approach
    can be used for them, as long as they abide by the required properties:

        * Identity Function: Computes the identities of given data points. Can be
        replaced with a prediction function.
        * Error Function: Computes the reconstruction/prediction error of one or
        multiple samples to a scalar (each).
        * Threshold Model: Maps the scalar to binary class labels.

    Note this kind of model *absolutely requires* a time series in most cases and
    therefore data passed to it, must be in order!
    """

    _if: FederatedModel
    _tm: FederatedModel
    _ef: Callable[[Tensor, Tensor], Tensor]

    _param_split: int

    _pf_mode: bool
    _prev_fit_sample: Optional[Tensor]
    _prev_pred_sample: Optional[Tensor]

    def __init__(
        self,
        identify_fn: FederatedModel,
        threshold_m: FederatedModel,
        error_fn: Callable[[Tensor, Tensor], Tensor],
        pf_mode: bool = False,
    ):
        """Creates a new federated IFTM anomaly detection model.

        :param identify_fn: Federated identity function model.
        :param threshold_m: Federated threshold model.
        :param error_fn: Reconstruction/Prediction error function that must compute
        the error in sample-wise manner (for example, if using a loss function,
        the reduction should be set to NONE).
        :param pf_mode: Whether IFTM uses an identity function or a prediction function.
        """
        self._if = identify_fn
        self._tm = threshold_m
        self._ef = error_fn

        self._param_split = len(identify_fn.get_parameters())

        self._pf_mode = pf_mode
        self._prev_fit_sample = None
        self._prev_pred_sample = None
        self._run_counter = 0

    def set_parameters(self, parameters: list[np.ndarray]):
        """Updates the internal parameters of the two underlying models by splitting
        the parameter lists as previously defined.

        :param parameters: Parameters to update the IFTM model with.
        """
        self._if.set_parameters(parameters[: self._param_split])
        self._tm.set_parameters(parameters[self._param_split :])

    def get_parameters(self) -> list[np.ndarray]:
        """Retrieves the weights of the underlying models.

        :return: Concatenated weight lists of the two models.
        """
        params = self._if.get_parameters()
        params.extend(self._tm.get_parameters())
        return params

    def fit2(self, x_data, y_data=None):
        """Trains the IFTM model with the given data by calling the wrapped models;
        first the IF to make a prediction, after which the error can be computed for
        the fitting of the TM. Afterward, the IF is fitted. Note that one can run
        IFTM in supervised mode by providing the true classes of each sample ---
        however most TMs require only the input data.

        Note that in case of an underlying prediction function (instead of a regular
        IF), the window is shifted by one step into the past, i.e. the final sample
        is only used to compute a prediction error, but not make a prediction,
        and it is stored for the next fitting step.

        Note this kind of model *absolutely requires* a time series in most cases and
        therefore data passed to it, must be in order! Also for the first step in a
        time series, it is impossible to compute a prediction error, since there is
        no previous sample to compare it to.

        :param x_data: Input data.
        :param y_data: Expected output, optional since default IFTM is fully
        unsupervised.
        """
        # Adjust input data depending on mode
        if self._pf_mode:
            x_data, y_true = self._shift_batch_window(x_data, fit=True)
            if x_data is None:
                return
        else:
            y_true = np.zeros(x_data.shape[0])  # x_data

        if self._run_counter == 0:
            self._run_counter += 1
            self._if.fit(x_data, y_true)
            y_pred = self._if.predict(x_data)
            y_detection = self._tm.predict(y_pred).astype(int)

        elif self._run_counter <= 4:
            self._run_counter += 1
            # Train TM
            y_pred = self._if.predict(x_data)
            # Train IF
            y_detection = self._tm.predict(y_pred).astype(int)

            if np.any(y_detection != 0):
                # Maske für Samples mit Fehler unterhalb des Schwellenwerts
                mask = y_detection != 0
                # Gefilterte X-Daten da gute daten 0 und schlechte 1 sind muss invertiert werden
                x_data_ok = x_data[tf.logical_not(mask)]
                y_true_ok = y_true[tf.logical_not(mask)]

                self._if.fit(x_data_ok, y_true_ok)
            else:
                self._if.fit(x_data, y_true)
        else:
            # Train TM
            y_pred = self._if.predict(x_data)
            pred_errs = self._ef(y_true, y_pred)

            mittelwert = np.mean(pred_errs)
            print(f"<<<<<<<<<<Fehlermittelwert: {mittelwert}")

            thresh = self._tm.get_threshold()
            print(f"Set threshold: {thresh:.4f}")

            # for test
            y_detection_tens = self._tm.predict(pred_errs)
            y_detection = y_detection_tens.numpy().astype(int).reshape(-1, 1)

        self.evaluate2(y_data, y_detection)

    def fit(self, x_data, y_data=None):
        """Trains the IFTM model with the given data by calling the wrapped models;
        first the IF to make a prediction, after which the error can be computed for
        the fitting of the TM. Afterward, the IF is fitted. Note that one can run
        IFTM in supervised mode by providing the true classes of each sample ---
        however most TMs require only the input data.

        Note that in case of an underlying prediction function (instead of a regular
        IF), the window is shifted by one step into the past, i.e. the final sample
        is only used to compute a prediction error, but not make a prediction,
        and it is stored for the next fitting step.

        Note this kind of model *absolutely requires* a time series in most cases and
        therefore data passed to it, must be in order! Also for the first step in a
        time series, it is impossible to compute a prediction error, since there is
        no previous sample to compare it to.

        :param x_data: Input data.
        :param y_data: Expected output, optional since default IFTM is fully
        unsupervised.
        """
        # Adjust input data depending on mode
        if self._pf_mode:
            x_data, y_true = self._shift_batch_window(x_data, fit=True)
            if x_data is None:
                return
        else:
            y_true = x_data

        # Überprüfen, ob y leer ist, und es ggf. mit Nullen auffüllen
        if y_data.size == 0:
            y_data = np.zeros(x_data.shape[0])

        if self._run_counter == 0:
            self._run_counter += 1
            self._if.fit(x_data, y_true)
            y_pred = self._if.predict(x_data)
            pred_errs = self._ef(y_true, y_pred)

            mittelwert = np.mean(pred_errs)
            print(f"<<<<<<<<<<Fehlermittelwert: {mittelwert}")

            # Compute errors and determine threshold using final labels
            self._tm.fit(pred_errs, y_data)
            thresh = self._tm.get_threshold()
            print(f"Set threshold: {thresh:.4f}")

            # for test
            y_detection_tens = self._tm.predict(pred_errs)
            y_detection = y_detection_tens.numpy().astype(int).reshape(-1, 1)
        elif self._run_counter <= 4:
            self._run_counter += 1
            # Train TM
            y_pred = self._if.predict(x_data)
            pred_errs = self._ef(y_true, y_pred)

            mittelwert = np.mean(pred_errs)
            print(f"<<<<<<<<<<Fehlermittelwert: {mittelwert}")

            thresh = self._tm.get_threshold()
            print(f"Set threshold: {thresh:.4f}")

            # for test
            y_detection_tens = self._tm.predict(pred_errs)
            y_detection = y_detection_tens.numpy().astype(int).reshape(-1, 1)

            # ein paar anomalien lernen
            # Finden der Indizes der Nullen
            zero_indices = np.where(y_detection == 0)[0]

            # Anzahl der zu ändernden Werte (20% der Nullen)
            num_changes = int(len(zero_indices) * 0.4)

            # Zufällige Auswahl von Indizes der Nullen
            change_indices = np.random.choice(zero_indices, num_changes, replace=False)

            # Ändere die ausgewählten 0en zu 1en
            y_detection_add = y_detection.copy()
            y_detection_add[change_indices] = 1

            # Gefilterte X-Daten da gute daten 0 und schlechte 1 sind muss invertiert werden
            x_data_below_threshold = x_data[y_detection_add.flatten() == 0]
            y_true_below_threshold = y_true[y_detection_add.flatten() == 0]

            self._tm.fit(pred_errs, y_data)
            # Train IF
            if len(x_data_below_threshold) > 2:
                # Compute errors and determine threshold using final labels
                self._if.fit(x_data_below_threshold, y_true_below_threshold)

        else:
            # Train TM
            y_pred = self._if.predict(x_data)
            pred_errs = self._ef(y_true, y_pred)

            mittelwert = np.mean(pred_errs)
            print(f"<<<<<<<<<<Fehlermittelwert: {mittelwert}")

            thresh = self._tm.get_threshold()
            print(f"Set threshold: {thresh:.4f}")

            # for test
            y_detection_tens = self._tm.predict(pred_errs)
            y_detection = y_detection_tens.numpy().astype(int).reshape(-1, 1)

        self.evaluate(y_data, y_detection, thresh, pred_errs)

    def evaluate(self, y_real, y_pred, thresh, pred_errs):
        try:
            y_real = y_real.squeeze()
            y_pred = y_pred.squeeze()

            if np.any(y_real != 0):
                print("jetzt")

            # Metrics
            cm = confusion_matrix(y_real, y_pred)
            roc = roc_auc_score(y_real, y_pred)
            prec, rec, _ = precision_recall_curve(y_real, y_pred)
            pr = auc(rec, prec)

            # Plots
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)

            # Gesamten Wertebereich bestimmen (nur falls Fehler vorhanden sind)
            all_errors = pred_errs[np.isin(y_real, [0, 1])]
            if len(all_errors.numpy().tolist()) > 0:
                min_val = min(all_errors.numpy().tolist())
                max_val = max(all_errors.numpy().tolist())
            else:
                min_val, max_val = 0, 1  # Fallback, falls keine Daten

            # Anzahl der Bins definieren (z.B. 10)
            num_bins = 10

            # Bins anhand des gemeinsamen Bereichs erstellen
            bins = np.linspace(min_val - 0.1, max_val + 0.1, num_bins + 1)

            # Histogramm für Normaldaten
            if len(pred_errs[y_real == 0].numpy().tolist()) > 0:
                sns.histplot(
                    pred_errs[y_real == 0].numpy().tolist(),
                    label="Normal",
                    kde=True,
                    bins=bins,
                )

            # Histogramm für Anomaliedaten
            if len(pred_errs[y_real == 1].numpy().tolist()) > 0:
                sns.histplot(
                    pred_errs[y_real == 1].numpy().tolist(),
                    label="Anomaly",
                    kde=True,
                    bins=bins,
                )

            plt.axvline(
                thresh, color="r", linestyle="--", label=f"Threshold={thresh:.3f}"
            )
            plt.xlabel("Reconstruction Error")  # X-Achse: Fehlerwerte
            plt.ylabel("Frequency")  # Y-Achse: Häufigkeit der Fehlerwerte
            plt.legend(loc="upper right")
            plt.title("Error Distribution")

            plt.subplot(1, 2, 2)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"],
            )
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.show()

            print(f"ROC AUC: {roc:.4f}, PR AUC: {pr:.4f}")
            return {"confusion_matrix": cm, "roc_auc": roc, "pr_auc": pr}

        except Exception as e:
            print(f"Error: {e}")
            print(f"y_test: {y_real}")
            print(f"errors if: {pred_errs}")
            print(f"y_pred tm: {y_pred}")

    def evaluate2(self, y_real, y_pred):
        try:
            y_real = y_real.squeeze()
            y_pred = y_pred.squeeze()

            if np.any(y_real != 0):
                print("jetzt")

            # Metrics
            cm = confusion_matrix(y_real, y_pred)
            roc = roc_auc_score(y_real, y_pred)
            prec, rec, _ = precision_recall_curve(y_real, y_pred)
            pr = auc(rec, prec)

            # Plots
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"],
            )
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.show()

            print(f"ROC AUC: {roc:.4f}, PR AUC: {pr:.4f}")
            return {"confusion_matrix": cm, "roc_auc": roc, "pr_auc": pr}

        except Exception as e:
            print(f"Error: {e}")
            print(f"y_test: {y_real}")
            print(f"y_pred tm: {y_pred}")

    def predict(self, x_data) -> Optional[Tensor]:
        """Makes a prediction on the given data and returns it by calling the wrapped
        models; first the IF to make a prediction, after which the error can be
        computed for the final prediction step using the TM.

        Note that in case of an underlying prediction function (instead of a regular
        IF), the window is shifted by one step into the past, i.e. the final sample
        is only used to compute a prediction error, but not make a prediction,
        and it is stored for the next prediction step.

        Note this kind of model *absolutely requires* a time series in most cases and
        therefore data passed to it, must be in order! Also for the first step in a
        time series, there is no previous sample to compare it to, therefore the
        sample is automatically classified as the default class (0, i.e. normal)

        :param x_data: Input data.
        :return: Predicted output tensor consisting of bools (0: normal, 1: abnormal).
        """
        # Adjust input data depending on mode
        if self._pf_mode:
            x_data, y_true = self._shift_batch_window(x_data, fit=False)
            if x_data is None:
                return tf.zeros(1)
        else:
            y_true = x_data
        # Make predictions
        y_pred = self._if.predict(x_data)
        pred_errs = self._ef(y_true, y_pred)
        return self._tm.predict(pred_errs)

    def _shift_batch_window(
        self, x_data, fit: bool
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Shifts a given input batch one step in the past, discarding the last
        sample from the batch and storing it for later user, but adding the last
        sample from the previous batch to the beginning of the batch. This is
        necessary for fitting and prediction of prediction-based IFs (see fit and
        predict function).

        :param x_data: Input data.
        :param fit: Whether the window is shifted for fitting or prediction purposes.
        :return: Shifted input data and adjusted test data.
        """
        if fit:
            prev_sample = self._prev_fit_sample
            self._prev_fit_sample = x_data[-1]
        else:
            prev_sample = self._prev_pred_sample
            self._prev_pred_sample = x_data[-1]

        if prev_sample is None:
            # first step of time series
            if len(x_data) == 1:
                # sample will be used later
                return None, None
            else:
                # decrease window to allow computation of errors
                y_true = x_data[1:]
                x_data = x_data[:-1]
        else:
            y_true = x_data
            x_data = tf.concat([prev_sample, x_data[:-1]], 0)
        return x_data, y_true


# def fit(self, x_data, y_data=None):
#    """Trains the IFTM model with the given data by calling the wrapped models;
#    first the IF to make a prediction, after which the error can be computed for
#    the fitting of the TM. Afterward, the IF is fitted. Note that one can run
#    IFTM in supervised mode by providing the true classes of each sample ---
#    however most TMs require only the input data.

#    Note that in case of an underlying prediction function (instead of a regular
#    IF), the window is shifted by one step into the past, i.e. the final sample
#    is only used to compute a prediction error, but not make a prediction,
#    and it is stored for the next fitting step.

#    Note this kind of model *absolutely requires* a time series in most cases and
#    therefore data passed to it, must be in order! Also for the first step in a
#    time series, it is impossible to compute a prediction error, since there is
#    no previous sample to compare it to.

#    :param x_data: Input data.
#    :param y_data: Expected output, optional since default IFTM is fully
#    unsupervised.
#    """
#    # Adjust input data depending on mode
#    if self._pf_mode:
#        x_data, y_true = self._shift_batch_window(x_data, fit=True)
#        if x_data is None:
#            return
#    else:
#        y_true = x_data

#    # Train TM
#    y_pred = self._if.predict(x_data)
#    pred_errs = self._ef(y_true, y_pred)

#    # reconstruction_errors = np.mean(np.square(y_true - y_pred), axis=1)

#    mittelwert = np.mean(pred_errs)
#    print(f"<<<<<<<<<<Fehlermittelwert: {mittelwert}")

#    # Compute errors and determine threshold using final labels
#    self._tm.fit(pred_errs, y_data)
#    thresh = self._tm.get_threshold()
#    print(f"Set threshold: {thresh:.4f}")

#    # if nur mit gutdaten tranineren
#    # Maske für Samples mit Fehler unterhalb des Schwellenwerts
#    mask = pred_errs > thresh
#    # Gefilterte X-Daten da gute daten 0 und schlechte 1 sind muss invertiert werden
#    x_data_below_threshold = x_data[tf.logical_not(mask)]
#    y_true_below_threshold = y_true[tf.logical_not(mask)]

#    # Train IF
#    if len(x_data_below_threshold) > 2:
#        self._if.fit(x_data_below_threshold, y_true_below_threshold)

#    # for test
#    y_detection_tens = self._tm.predict(pred_errs)
#    y_detection = y_detection_tens.numpy().astype(int).reshape(-1, 1)

#    # Überprüfen, ob y leer ist, und es ggf. mit Nullen auffüllen
#    if y_data.size == 0:
#        y_data = np.zeros(x_data.shape[0])

#    self.evaluate(y_data, y_detection, thresh, pred_errs)
