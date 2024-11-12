from keras import layers, models


def create_cnn(architecture: int):
    match architecture:
        case 0:
            return models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(128, activation='relu'),
                layers.Dense(10, activation='softmax')
            ])
        case 1:
            return models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(128, activation='relu'),
                layers.Dense(10, activation='softmax')
            ])


def create_autoencoder(architecture: int):
    match architecture:
        case 0:
            return models.Sequential([
                layers.Input(shape=(2,)),
                layers.Dense(2),
                layers.Dense(1),
                layers.Dense(2),
            ])
        case 1:
            return models.Sequential([
                layers.Input(shape=(2,)),
                layers.Dense(2),
                layers.Dense(1),
                layers.Dense(1),
                layers.Dense(2),
            ])
        case 2:
            return models.Sequential([
                layers.Input(shape=(5,)),
                layers.Dense(5),
                layers.Dense(4),
                layers.Dense(2),
                layers.Dense(1),
                layers.Dense(4),
                layers.Dense(5),
            ])
        case 3:
            return models.Sequential([
                layers.Input(shape=(3,)),
                layers.Dense(3),
                layers.Dense(2),
                layers.Dense(1),
                layers.Dense(2),
                layers.Dense(3),
            ])
        case 4:
            return models.Sequential([
                layers.Input(shape=(6,)),
                layers.Dense(5),
                layers.Dense(4),
                layers.Dense(3),
                layers.Dense(3),
                layers.Dense(4),
                layers.Dense(5),
                layers.Dense(6)
            ])
        case 5:
            return models.Sequential([
                layers.Input(shape=(6,)),
                layers.Dense(4),
                layers.Dense(3),
                layers.Dense(3),
                layers.Dense(4),
                layers.Dense(6),
            ])
        case 99:
            return models.Sequential([
                layers.Input(shape=(65,)),
                layers.Dense(32),
                layers.Dense(16),
                layers.Dense(8),
                layers.Dense(4),
                layers.Dense(2),
                layers.Dense(1)
            ])
