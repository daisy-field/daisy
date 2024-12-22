import os.path
import numpy
import tensorflow as tf
from data_processor import get_dataset
from test_models import create_autoencoder
from keras.optimizers import Optimizer, Adam
from keras.losses import Loss, MeanAbsoluteError, BinaryCrossentropy
from keras.models import Sequential
from keras.metrics import Precision, Recall, F1Score
from keras.callbacks import Callback

class LossLogger(Callback):
    def on_train_batch_end(self, batch, logs=None):
        tf.print(self.model.input)

def train(train_model: Sequential, train_data: numpy.ndarray,
          train_labels: numpy.ndarray, train_loss: Loss, train_optimizer: Optimizer, model_num: int):
    train_model.compile(optimizer=train_optimizer,
                        loss=train_loss,
                        metrics=['accuracy', Precision(thresholds=0.5), Recall(thresholds=0.5), F1Score(threshold=0.5)])
    path = f'./testing_weights/ids_auto_{model_num}.ckpt'
    if not os.path.isfile(f'{path}.data-00000-of-00001'):
        train_model.fit(train_data, train_labels, epochs=10, validation_split=0.2, batch_size=32)
        train_model.save_weights(path)


if __name__ == "__main__":
    # Get dataset
    data, labels = get_dataset()

    # Creating the model
    print("Getting model...")
    model_num = 5
    model = create_autoencoder(model_num)

    # Training the model
    print("Training model...")
    loss = BinaryCrossentropy()
    optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
    train(model, data, labels, loss, optimizer, model_num)
