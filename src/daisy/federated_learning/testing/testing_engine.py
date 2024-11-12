import os.path
import numpy
from data_processor import get_dataset
from test_models import create_autoencoder
from keras.optimizers import Optimizer, Adam
from keras.losses import Loss, MeanSquaredError
from keras.models import Sequential

def train(train_model: Sequential, train_data: numpy.ndarray,
          train_labels: numpy.ndarray, train_loss: Loss, train_optimizer: Optimizer, model_num: int):
    train_model.compile(optimizer=train_optimizer,
                        loss=train_loss,
                        metrics=['accuracy'])
    path = f'./testing_weights/ids_auto_{model_num}.ckpt'
    if not os.path.isfile(f'./testing_weights/ids.ckpt'):
        train_model.fit(train_data, train_labels, epochs=10, validation_split=0.2, batch_size=32)
        train_model.save_weights(path)
    else:
        train_model.load_weights(path)

if __name__ == "__main__":
    # Get dataset
    print("Getting data...")
    data, labels = get_dataset()
    print("Got dataset of shape " + str(data.shape))
    print("Got label of shape " + str(labels.shape))

    # Creating the model
    print("Getting model...")
    model_type = 99
    model = create_autoencoder(model_type)

    # Training the model
    print("Training model...")
    loss = MeanSquaredError()
    optimizer = Adam(learning_rate=0.0001)
    train(model, data, labels, loss, optimizer, model_type)
