import os.path
import csv
from data_processor import get_dataset
from test_models import create_model
from keras.models import Sequential
from keras.losses import BinaryCrossentropy
from keras.metrics import Precision, Recall, F1Score
from src.daisy.federated_learning.federated_aggregator import LCAggregator


def load_model(test_model: Sequential, model_num: int, dataset: str):
    path = f'./testing_weights/CIC_{dataset}/ids_auto_{model_num}.ckpt'

    test_model.compile(loss=BinaryCrossentropy(),
                       metrics=['accuracy', Precision(thresholds=0.5), Recall(thresholds=0.5), F1Score(threshold=0.5)])

    if os.path.isfile(f'{path}.data-00000-of-00001'):
        print("Loading pre-trained weights...")
        test_model.load_weights(path)
    else:
        print("No pre-trained weights available!")


if __name__ == "__main__":
    model_nums = [6,5,7,9]
    # Get dataset
    test_data, test_labels = get_dataset(True, 2)

    # Creating the model and loading the weights
    print("Getting models and loading weights...")
    models = [create_model(num) for num in model_nums]
    for (i, num) in enumerate(model_nums):
        load_model(models[i], num, '1' if i == 0 else '2')

    # Evaluate the models
    print("Evaluating the models...")
    solo_scores = [model.evaluate(test_data, test_labels) for model in models]
    # Aggregate
    print("Aggregating the models...")
    aggregator = LCAggregator(dict(zip(model_nums, [model.layers for model in models])))
    aggregation_result = aggregator.aggregate(
        (model_nums[0], models[0].get_weights()),
        list(zip(model_nums[1:], [model.get_weights() for model in models[1:]])))

    models[0].set_weights(aggregation_result)
    aggr_score = models[0].evaluate(test_data, test_labels)

    # Write results to file
    file_name = '_'.join(str(num) for num in model_nums)
    with open(f"{os.getcwd()}/results/{file_name}.csv", mode="w+", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        for score in solo_scores:
            writer.writerow(score)
        writer.writerow(aggr_score)
