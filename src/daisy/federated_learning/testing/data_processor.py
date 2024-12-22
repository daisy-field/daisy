import pandas as pd
import numpy as np

def get_dataset() -> tuple[np.ndarray, np.ndarray]:
    # Load dataset
    print("Getting data...")
    df = pd.read_csv('combined.csv', header=None)
    data = df.iloc[:, :-1].to_numpy()
    labels = df.iloc[:, -1].clip(0, 1).to_numpy(dtype=np.float32)

    print(np.isnan(data).any())
    print(np.unique(labels))

    print("Got dataset of shape " + str(data.shape))
    print("Got label of shape " + str(labels.shape))
    return data, labels