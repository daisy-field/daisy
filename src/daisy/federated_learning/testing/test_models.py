import tensorflow as tf
from keras import layers, models
from keras.activations import sigmoid


def create_autoencoder(architecture: int):
    match architecture:
        case 0:
            return models.Sequential([
                layers.Input(shape=(65,)),
                layers.Dense(32),
                layers.Dense(16),
                layers.Dense(8),
                layers.Dense(4),
                layers.Dense(2),
                layers.Dense(1, activation=sigmoid)
            ])
        case 1:
            return models.Sequential([
                layers.Input(shape=(65,)),
                layers.Reshape((1, 65)),
                layers.LSTM(64, return_sequences=False),

                layers.Dense(64),
                layers.Dense(32),
                # layers.Dense(16),
                layers.Dense(8),
                layers.Dense(4),
                layers.Dense(2),
                layers.Dense(1, activation=sigmoid)
            ])
        case 2:
            return models.Sequential([
                layers.Input(shape=(65,)),
                layers.Dense(48),
                layers.Dense(32),
                layers.Dense(16),
                layers.Dense(8),
                layers.Dense(4),
                layers.Dense(2),
                layers.Dense(1, activation=sigmoid)
            ])
        case 3:
            return models.Sequential([
                layers.Input(shape=(65,)),
                layers.Dense(48),
                layers.Dense(32),
                layers.Dense(8),
                layers.Dense(2),
                layers.Dense(1, activation=sigmoid)
            ])
        case 4:
            return models.Sequential([
                layers.Input(shape=(65,)),
                layers.Reshape((1, 65)),
                layers.LSTM(64, return_sequences=False),
                layers.Dense(64),
                layers.Dense(32),
                layers.Dense(8),
                layers.Dense(2),
                layers.Dense(1, activation=sigmoid)
            ])

        # CICIDS MODELS FROM HERE ON
        case 5:
            return models.Sequential([
                layers.Input(shape=(78,)),
                layers.Dense(40),
                layers.Dense(20),
                layers.Dense(10),
                layers.Dense(5),
                layers.Dense(2),
                layers.Dense(1, activation=sigmoid),
                layers.Lambda(lambda x: tf.clip_by_value(x, 0.4, 0.6))
            ])
        case 6:
            return models.Sequential([
                layers.Input(shape=(78,)),
                layers.Reshape((1, 78)),
                layers.LSTM(78, return_sequences=False),

                layers.Dense(78),
                layers.Dense(40),
                layers.Dense(10),
                layers.Dense(5),
                layers.Dense(2),
                layers.Dense(1, activation=sigmoid)
            ])
        case 7:
            return models.Sequential([
                layers.Input(shape=(78,)),
                layers.Dense(60),
                layers.Dense(40),
                layers.Dense(20),
                layers.Dense(10),
                layers.Dense(5),
                layers.Dense(2),
                layers.Dense(1, activation=sigmoid)
            ])
        case 8:
            return models.Sequential([
                layers.Input(shape=(78,)),
                layers.Dense(60),
                layers.Dense(40),
                layers.Dense(10),
                layers.Dense(2),
                layers.Dense(1, activation=sigmoid)
            ])
        case 9:
            return models.Sequential([
                layers.Input(shape=(78,)),
                layers.Reshape((1, 78)),
                layers.LSTM(78, return_sequences=False),
                layers.Dense(78),
                layers.Dense(40),
                layers.Dense(10),
                layers.Dense(2),
                layers.Dense(1, activation=sigmoid)
            ])
