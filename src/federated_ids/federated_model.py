import keras
from tensorflow.keras.models import Sequential
input_size = 70


class FederatedModel:
    """Creates the keras model for the federated training"""

    @staticmethod
    def build():
        encoder = Sequential([
            keras.layers.Dense(input_size, input_shape=(input_size,)),
            keras.layers.Dense(35),
            keras.layers.Dense(18),
        ])
        decoder = Sequential([
            keras.layers.Dense(35, input_shape=(18,)),
            keras.layers.Dense(input_size),
            keras.layers.Activation("sigmoid"),
        ])
        return encoder, decoder
