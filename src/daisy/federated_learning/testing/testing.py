import keras.losses
import os
import numpy as np
import test_models as tm
from src.daisy.federated_learning.federated_aggregator import LCAggregator, FedAvgAggregator
from keras import datasets


def train_model(model, model_num, path, loss, train_data, train_labels, val_data=None, val_labels=None):
    path = path.format(model_num=model_num)
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])
    if not os.path.isfile(f'{path}.data-00000-of-00001'):
        if val_data is None or val_labels is None:
            model.fit(train_data, train_labels, epochs=10, validation_split=0.2, batch_size=16)
        else:
            model.fit(train_data, train_labels, epochs=10,
                      validation_data=(val_data, val_labels))
        model.save_weights(path)
    else:
        model.load_weights(path)


model_type = 'cnn'

match model_type:
    case 'cnn':
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

        path = './trained/cnn_{model_num:04d}.ckpt'
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=path,
            verbose=1,
            save_freq='epoch'
        )
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        model = tm.create_cnn(0)
        train_model(model, 0, path, loss,
                    train_images, train_labels, test_images, test_labels)

        model2 = tm.create_cnn(1)
        train_model(model2, 1, path, loss,
                    train_images, train_labels, test_images, test_labels)

        model.evaluate(test_images, test_labels)

        aggregator = LCAggregator({0: model.layers, 1: model2.layers})
        aggregation_result = aggregator.aggregate([(0, model.get_weights()), (1, model2.get_weights())])

        model.set_weights(aggregation_result[0])
        model.evaluate(test_images, test_labels)

    case 'auto':
        train_data = np.random.rand(1000, 2)
        train_data2 = np.random.rand(1000, 3)
        bad_model_data = np.random.rand(1000, 5)

        path = './trained/auto_{model_num:04d}.ckpt'
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=path,
            verbose=1,
            save_freq='epoch'
        )
        loss = 'mse'

        model = tm.create_autoencoder(0)
        train_model(model, 0, path, loss,
                    train_data, train_data)

        model2 = tm.create_autoencoder(1)
        train_model(model2, 1, path, loss,
                    train_data, train_data)

        model3 = tm.create_autoencoder(3)
        train_model(model3, 3, path, loss,
                    train_data2, train_data2)

        test_data = np.random.rand(1000, 2)
        model.evaluate(test_data, test_data)

        aggregator = LCAggregator({0: model.layers, 1: model2.layers, 2: model3.layers})
        aggregation_result = aggregator.aggregate(
            [(0, model.get_weights()), (1, model2.get_weights()), (2, model3.get_weights())])

        model.set_weights(aggregation_result[0])
        model.evaluate(test_data, test_data)
