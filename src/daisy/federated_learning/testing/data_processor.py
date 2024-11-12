import pandas as pd
import numpy as np


def get_dataset() -> tuple[np.ndarray, np.ndarray]:
    # Load dataset
    df = pd.read_csv('test_pcap.csv', header=None)
    data = df.iloc[:, :-1]
    labels = df.iloc[:, -1]
    return data.to_numpy(), labels.to_numpy()