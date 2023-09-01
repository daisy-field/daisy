"""
    A collection of various types of model wrappers, implementing the same interface for each federated model type, thus
    enabling their inter-compatibility for different aggregation strategies and federated system components.

    Author: Fabian Hofmann, Seraphin Zunzer
    Modified: 14.08.23

    TODO Future Work should be the implementation of Open Source Interfaces (e.g. Keras Model API)
"""
from abc import ABC, abstractmethod
from typing import Callable

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
    def fit(self, x_data, y_data, *args, **kwargs):
        """Trains the model with the given data, which must be compatible with the tensorflow API
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).

        :param x_data: Input data.
        :param y_data: Expected output.
        :param args: Arguments. Depends on the type of model to be implemented.
        :param kwargs: Keyword arguments. Depends on the type of model to be implemented.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_data, *args, **kwargs) -> Tensor:
        """Makes a prediction on the given data and returns it, which must be compatible with the tensorflow API
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).

        :param x_data: Input data.
        :param args: Arguments. Depends on the type of model to be implemented.
        :param kwargs: Keyword arguments. Depends on the type of model to be implemented.
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

    def fit(self, x_data, y_data, *args, **kwargs):
        """Trains the model with the given data by calling the wrapped model, which must be compatible with the
        tensorflow API (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).

        :param x_data: Input data.
        :param y_data: Expected output.
        :param args: Not supported arguments.
        :param kwargs: Not supported keyword arguments.
        """
        self._model.fit(x=x_data, y=y_data, batch_size=self._batch_size, epochs=self._epochs)

    def predict(self, x_data, *args, **kwargs) -> Tensor:
        """Makes a prediction on the given data and returns it by calling the wrapped model. Uses the call() tensorflow
        model interface for small numbers of data points, which must be compatible with the tensorflow API
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).

        :param x_data: Input data.
        :param args: Not supported arguments.
        :param kwargs: Not supported keyword arguments.
        :return: Predicted output tensor.
        """
        if len(x_data) > self._batch_size:
            return self._model.predict(x=x_data, batch_size=self._batch_size)
        return self._model(x_data, training=False).numpy()


# TODO WRAP TM AS EXTENDED TF MODEL CLASS

class IFTMFederatedModel(FederatedModel):
    """Double union of two federated models, following the IFTM hybdrid  model approach --- identify function threshold
    model principle by Florian et al. (https://ieeexplore.ieee.org/document/8456348): One for the computation of the
    identity of a given sample (alternatively prediction of the next sample), while the other maps the error/loss using
    a threshold(-model) to the binary class labels for anomaly detection. Both are generic federated models, so any
    approach can be used for them, as long as they abide by the required properties:

        * Identity Function: Computes the identities of given data points. Can be replaced with a prediction function.
        * Threshold Model: Maps the predicted vs actual value from the IF to a scalar, then maps it to a binary label.
    """
    _if: FederatedModel
    _tm: FederatedModel

    _param_split: int

    def __init__(self, identify_fn: FederatedModel, threshold_m, param_split: int):
        """Creates a new federated IFTM anomaly detection model.

        :param identify_fn: Federated identity function model.
        :param threshold_m: Federated threshold model.
        :param param_split: Length of IF parameters to efficiently merge the two lists of params.
        """
        self._if = identify_fn
        self._tm = threshold_m

        self._param_split = param_split

    def set_parameters(self, parameters: list[np.ndarray]):
        """Updates the internal parameters of the model.

        :param parameters: Parameters to update the IFTM model with.
        """
        self._if.set_parameters(parameters[:self._param_split])
        self._tm.set_parameters(parameters[self._param_split:])

    def get_parameters(self) -> list[np.ndarray]:
        """Retrieves the weights of the underlying model.
TODO
        :return:
        """
        params = self._if.get_parameters()
        params.extend(self._tm.get_parameters())
        return params

    def fit(self, x_data, y_data, *args, **kwargs):
        """Trains the model with the given data by calling the wrapped model, which must be compatible with the
        tensorflow API (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).
        TODO
        :param x_data:
        :param y_data:
        :param args: Not supported arguments.
        :param kwargs: Not supported keyword arguments.
        """
        pass

    def predict(self, x_data, y_data=None, *args, **kwargs) -> Tensor:
        """Makes a prediction on the given data and returns it by calling the wrapped model. Uses the call() tensorflow
        model interface for small numbers of data points, which must be compatible with the tensorflow API
        (see: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).
TODO
        :param x_data:
        :param y_data:
        :param args: Not supported arguments.
        :param kwargs: Not supported keyword arguments.
        :return:
        """
        y_pred = self._if.predict(x_data)
        self._tm.predict(y_pred, y_data) # FIXME separate error function from TM, this is just a mess and makes everything worse



def get_tf_error_fn(tf_metric: keras.metrics.Metric) -> Callable[[Tensor, Tensor], Tensor]:
    """
TODO
    :param tf_metric:
    :return:
    """
    return lambda t_label, p_label: tf_metric(t_label, p_label)

# """ FIXME MUST BE MADE COMPLIANT WITH FED MODEL ABSTRACT CLASS
#     TODO CAN BE MOVED DIRECTLY INTO FEDERATED MODEL. PYTON IS NOT JAVA!
#     Federated autoencoder that implements federated_models interface.
#
#     Author: Seraphin Zunzer
#     Modified: 09.08.23
# """
# import logging
#
# import keras
# import tensorflow as tf
# from federated_learning.federated_model import FederatedModel
#
# input_size = 65
#
#
# # FIXME EVERYTHING
#
# # TODO MOVE TO FEDERATED MODEL
#
# class FedAutoencoder(FederatedModel):
#     """Class for federated autoencoder"""
#     _model = None
#
#     def __init__(self):
#         """
#         Build the autoencoder
#
#         :return: built model
#         """
#         encoder = tf.keras.models.Sequential([
#             keras.layers.Dense(input_size, input_shape=(input_size,)),
#             keras.layers.Dense(35),
#             keras.layers.Dense(18),
#         ])
#         decoder = tf.keras.models.Sequential([
#             keras.layers.Dense(35, input_shape=(18,)),
#             keras.layers.Dense(input_size),
#             keras.layers.Activation("sigmoid"),
#         ])
#         input_format = keras.layers.Input(shape=(input_size,))
#         self.model = tf.keras.models.Model(inputs=input_format, outputs=decoder(encoder(input_format)))
#         logging.info("Model created")
#
#     def init_model(self):
#         """
#         Compile the model for prediction
#
#         :return: compiled model
#         """
#         self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=[])
#         logging.info("Compiled Model")
#         return self.model
