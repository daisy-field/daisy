import keras.losses
import os
from federated_aggregator import LCAggregator
from keras import datasets, layers, models


def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def create_model2():
    return models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
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


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

path = './trained/{model_num:04d}.ckpt'
model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=path,
    verbose=1,
    save_weights_only=True,
    save_freq='epoch'
)

model = create_model()
if not os.path.isfile('./trained/0001.ckpt.data-00000-of-00001'):
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.save_weights(path.format(model_num=1))
    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))
else:
    model.load_weights(path.format(model_num=1))

model2 = create_model2()
if not os.path.isfile('./trained/0002.ckpt.data-00000-of-00001'):
    model2.compile(optimizer='adam',
                   loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
    model.save_weights(path.format(model_num=2))
    history2 = model.fit(train_images, train_labels, epochs=10,
                         validation_data=(test_images, test_labels))
else:
    model2.load_weights(path.format(model_num=2))

aggregator = LCAggregator({'0': model.layers, '1': model2.layers})
