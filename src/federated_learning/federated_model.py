"""
    A collection of various types of model wrappers, implementing the same interface for each federated model type, thus
    enabling their inter-compatibility for different aggregation strategies and federated system components.

    Author: Fabian Hofmann, Seraphin Zunzer
    Modified: 14.08.23

    TODO Future Work should be the implementation of Open Source Interfaces (e.g. Keras Model API)
"""
from abc import ABC, abstractmethod
from typing import Callable, Self

import numpy as np
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

    def __init__(self, model: keras.Model, optimizer: str | keras.optimizers, loss: str | keras.losses.Loss,
                 metrics: list[str | Callable, keras.metrics.Metric] = None,
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
    def get_fae(cls, input_size: int, optimizer: str | keras.optimizers = "Adam", loss: str | keras.losses.Loss = "mse",
                metrics: list[str | Callable, keras.metrics.Metric] = None,
                batch_size: int = 32, epochs: int = 1) -> Self:
        """Factory class method to create a simple federated autoencoder model of a fixed depth but with variable input
        size.

        Note this setup could also be created with an unsupervised federated model (see TODO

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
        encoder = keras.layers.Dense(18)(x)

        dec_inputs = keras.layers.Input(shape=(18,))
        y = keras.layers.Dense(35)(dec_inputs)
        y = keras.layers.Dense(input_size)(y)
        decoder = keras.layers.Activation("sigmoid")(y)

        ae = keras.models.Model(inputs=enc_inputs, outputs=decoder(encoder))
        return TFFederatedModel(ae, optimizer, loss, metrics, batch_size, epochs)

class UnsupervisedFederatedModel(FederatedModel):
    """TODO

    """
    pass


class IFTMFederatedModel(FederatedModel):
    """Double union of two federated models, following the IFTM hybdrid  model approach --- identify function threshold
    model principle by Florian et al. (https://ieeexplore.ieee.org/document/8456348): One for the computation of the
    identity of a given sample (alternatively prediction of the next sample), while the other maps the error/loss using
    a threshold(-model) to the binary class labels for anomaly detection. Both are generic federated models, so any
    approach can be used for them, as long as they abide by the required properties:

        * Identity Function: Computes the identities of given data points. Can be replaced with a prediction function.
        * Error Function: Computes the reconstruction/prediction error of one or multiple samples to a scalar (each).
        * Threshold Model: Maps the scalar to binary class labels.
    """
    _if: FederatedModel
    _tm: FederatedModel
    _ef: Callable[[Tensor, Tensor], Tensor]

    _param_split: int

    def __init__(self, identify_fn: FederatedModel, threshold_m: FederatedModel,
                 error_fn: Callable[[Tensor, Tensor], Tensor], param_split: int, pred_mode: bool = False):
        """Creates a new federated IFTM anomaly detection model.

        :param identify_fn: Federated identity function model.
        :param threshold_m: Federated threshold model.
        :param error_fn: Reconstruction/Prediction error function.
        :param param_split: Length of IF parameters to efficiently merge the two lists of params.
        """
        self._if = identify_fn
        self._tm = threshold_m
        self._ef = error_fn

        self._param_split = param_split

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

    def fit(self, x_data, _):
        """Trains the IFTM model with the given data by calling the wrapped models; first the IF to make a prediction,
        after which the error can be computed for the fitting of the TM. Afterward, the IF is fitted.

        Note that IFTM requires no y_data as it is entirely unsupervised --- the fitting happens using the input data
        only.

        :param x_data: Input data.
        :param _: Ignored expected output parameter.
        """
        y_pred = self._if.predict(x_data)
        pred_errs = self._ef(x_data, y_pred)
        self._tm.fit(pred_errs, None)

        self._if.fit(x_data, x_data)

    def predict(self, x_data) -> Tensor:
        """Makes a prediction on the given data and returns it bby calling the wrapped models; first the IF to make a
        prediction, after which the error can be computed for the final prediction step using the TM.

        :param x_data: Input data.
        :return: Predicted output tensor.
        """
        y_pred = self._if.predict(x_data)
        pred_errs = self._ef(x_data, y_pred)
        return self._tm.predict(pred_errs)

    @staticmethod
    def get_tf_error_fn(tf_metric: keras.metrics.Metric) -> Callable[[Tensor, Tensor], Tensor]:
        """Quick wrapper for tensorflow metric objects as error function for IFTM models.

        :param tf_metric: Tensorflow metric object to be wrapped.
        :return: Wrapped tensorflow metric object as callable function.
        """
        return lambda t_label, p_label: tf_metric(t_label, p_label)
