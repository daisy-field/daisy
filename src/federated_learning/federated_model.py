"""
    A collection of various types of model wrappers, implementing the same interface for each federated model type, thus
    enabling their inter-compatibility for different aggregation strategies and federated system components.

    Author: Fabian Hofmann
    Modified: 09.09.23

    TODO Future Work should be the implementation of Open Source Interfaces (e.g. Keras Model API)
"""
from abc import ABC, abstractmethod
from typing import Callable, Optional, Self

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow import keras


class FederatedModel(ABC):
    """An abstract model wrapper that offers the same methods, no matter the type of underlying model. Must always be
    implemented if a new model type is to be used in the federated system.
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
        """Trains the model with the given data, which must be compatible with the tensorflow API
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).

        :param x_data: Input data.
        :param y_data: Expected output.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_data) -> Tensor:
        """Makes a prediction on the given data and returns it, which must be compatible with the tensorflow API
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).

        :param x_data: Input data.
        :return: Predicted output tensor.
        """
        raise NotImplementedError


class TFFederatedModel(FederatedModel):
    """The standard federated model wrapper for tensorflow models. Can be used for both online and offline training/
    prediction, by default online, however.
    """
    _model: keras.Model
    _batch_size: int
    _epochs: int

    def __init__(self, model: keras.Model, optimizer: str | keras.optimizers.Optimizer, loss: str | keras.losses.Loss,
                 metrics: list[str | Callable | keras.metrics.Metric] = None,
                 batch_size: int = 32, epochs: int = 1):
        """Creates a new tensorflow federated model from a given model. Since this also compiles the given model,
        there are set of additional arguments, for more information on those see:
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile

        :param model: Underlying model to be wrapped around.
        :param optimizer: Optimizer to use during training.
        :param loss: Loss function to use during training.
        :param metrics: Evaluation metrics to be displayed during training and testing.
        :param batch_size: Batch size during training and prediction.
        :param epochs: Number of epochs (rounds) during training.
        """
        self._model = model
        self._model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

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
        """Trains the model with the given data by calling the wrapped model, which must be compatible with the
        tensorflow API (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).

        :param x_data: Input data.
        :param y_data: Expected output.
        """
        self._model.fit(x=x_data, y=y_data, batch_size=self._batch_size, epochs=self._epochs)

    def predict(self, x_data) -> Tensor:
        """Makes a prediction on the given data and returns it by calling the wrapped model. Uses the call() tensorflow
        model interface for small numbers of data points, which must be compatible with the tensorflow API
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).

        :param x_data: Input data.
        :return: Predicted output tensor.
        """
        if len(x_data) > self._batch_size:
            return self._model.predict(x=x_data, batch_size=self._batch_size)
        return self._model(x_data, training=False).numpy()

    @classmethod
    def get_fae(cls, input_size: int,
                optimizer: str | keras.optimizers.Optimizer = "Adam", loss: str | keras.losses.Loss = "mse",
                metrics: list[str | Callable | keras.metrics.Metric] = None,
                batch_size: int = 32, epochs: int = 1) -> Self:
        """Factory class method to create a simple federated autoencoder model of a fixed depth but with variable input
        size.

        Should only serve as a quick and basic setup for a model.

        :param input_size: Dimensionality of input (and therefore output) of autoencoder.
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


class FederatedIFTM(FederatedModel):
    """Double union of two federated models, following the IFTM hybdrid  model approach --- identify function threshold
    model principle by Schmidt et al. (https://ieeexplore.ieee.org/document/8456348): One for the computation of the
    identity of a given sample (alternatively prediction of the next sample), while the other maps the error/loss using
    a threshold(-model) to the binary class labels for anomaly detection. Both are generic federated models, so any
    approach can be used for them, as long as they abide by the required properties:

        * Identity Function: Computes the identities of given data points. Can be replaced with a prediction function.
        * Error Function: Computes the reconstruction/prediction error of one or multiple samples to a scalar (each).
        * Threshold Model: Maps the scalar to binary class labels.

    Note this kind of model *absolutely requires* a time series in most cases and therefore data passed to it, must be
    in order!
    """
    _if: FederatedModel
    _tm: FederatedModel
    _ef: Callable[[Tensor, Tensor], Tensor]

    _param_split: int

    _pf_mode: bool
    _prev_fit_sample: Optional[Tensor]
    _prev_pred_sample: Optional[Tensor]

    def __init__(self, identify_fn: FederatedModel, threshold_m: FederatedModel,
                 error_fn: Callable[[Tensor, Tensor], Tensor], param_split: int, pf_mode: bool = False):
        """Creates a new federated IFTM anomaly detection model.

        :param identify_fn: Federated identity function model.
        :param threshold_m: Federated threshold model.
        :param error_fn: Reconstruction/Prediction error function.
        :param param_split: Length of IF parameters to efficiently merge the two lists of params.
        :param pf_mode: Whether IFTM uses an identity function as IF or a prediction function.
        """
        self._if = identify_fn
        self._tm = threshold_m
        self._ef = error_fn

        self._param_split = param_split

        self._pf_mode = pf_mode
        self._prev_fit_sample = None
        self._prev_pred_sample = None

    def set_parameters(self, parameters: list[np.ndarray]):
        """Updates the internal parameters of the two underlying models by splitting the parameter lists as previously
        defined.

        :param parameters: Parameters to update the IFTM model with.
        """
        self._if.set_parameters(parameters[:self._param_split])
        self._tm.set_parameters(parameters[self._param_split:])

    def get_parameters(self) -> list[np.ndarray]:
        """Retrieves the weights of the underlying models.

        :return: Concatenated weight lists of the two models.
        """
        params = self._if.get_parameters()
        params.extend(self._tm.get_parameters())
        return params

    def fit(self, x_data, y_data=None):
        """Trains the IFTM model with the given data by calling the wrapped models; first the IF to make a prediction,
        after which the error can be computed for the fitting of the TM. Afterward, the IF is fitted. Note that one can
        run IFTM in supervised mode by providing the true classes of each sample --- however most TMs require only the
        input data.

        Note that in case of an underlying prediction function (instead of a regular IF), the window is shifted by one
        step into the past, i.e. the final sample is only used to compute a prediction error, but not make a prediction,
        and it is stored for the next fitting step.

        Note this kind of model *absolutely requires* a time series in most cases and therefore data passed to it, must
        be in order! Also for the first step in a time series, it is impossible to compute a prediction error, since
        there is no previous sample to compare it to.

        :param x_data: Input data.
        :param y_data: Expected output, optional since default IFTM is fully unsupervised.
        """
        # Adjust input data depending on mode
        if self._pf_mode:
            x_data, y_true = self._shift_batch_window(x_data, fit=True)
            if x_data is None:
                return
        else:
            y_true = x_data
        # Train TM
        y_pred = self._if.predict(x_data)
        pred_errs = self._ef(y_true, y_pred)
        self._tm.fit(pred_errs, y_data)
        # Train IF
        self._if.fit(x_data, y_true)

    def predict(self, x_data) -> Optional[Tensor]:
        """Makes a prediction on the given data and returns it bby calling the wrapped models; first the IF to make a
        prediction, after which the error can be computed for the final prediction step using the TM.

        Note that in case of an underlying prediction function (instead of a regular IF), the window is shifted by one
        step into the past, i.e. the final sample is only used to compute a prediction error, but not make a prediction,
        and it is stored for the next prediction step.

        Note this kind of model *absolutely requires* a time series in most cases and therefore data passed to it, must
        be in order! Also for the first step in a time series, there is no previous sample to compare it to, therefore
        the sample is automatically classified as the default class (0, i.e. normal)

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

    def _shift_batch_window(self, x_data, fit: bool) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Shifts a given input batch one step in the past, discarding the last sample from the batch and storing it for
        later user, but adding the last sample from the previous batch to the beginning of the batch. This is necessary
        for fitting and prediction of prediction-based IFs (see fit and predict function).

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

    @staticmethod
    def get_tf_error_fn(tf_metric: keras.metrics.Metric) -> Callable[[Tensor, Tensor], Tensor]:
        """Quick wrapper for tensorflow metric objects as error function for IFTM models.

        :param tf_metric: Tensorflow metric object to be wrapped.
        :return: Wrapped tensorflow metric object as callable function.
        """
        return lambda t_label, p_label: tf_metric(t_label, p_label)
