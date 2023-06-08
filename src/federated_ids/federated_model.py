import keras
from tensorflow.keras.models import Sequential

input_size = 70


class FederatedModel():
    """Creates the keras model for the federated training"""

    def build_autoencoder(self):
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

    @staticmethod
    def compile_model():
        """

        :return:
        """
        input_format = keras.layers.Input(shape=(input_size,))
        enc, dec = self.build_autoencoder()
        model = tf.keras.models.Model(inputs=input_format, outputs=dec(enc(input_format)))
        model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='mse', metrics=[])
        return model