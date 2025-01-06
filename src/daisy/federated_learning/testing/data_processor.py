import pandas as pd
import numpy as np

def get_dataset(split: bool = False, split_num: int = 0) -> tuple[np.ndarray, np.ndarray]:
    # Load dataset
    print("Getting data...")
    df = pd.read_csv('CIC-IDS17.csv', header=None)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if split:
        print("Original shape " + str(df.shape))
        label_data = df.to_numpy(np.float32)
        np.random.shuffle(label_data)

        if split_num == 1:
            label_data = label_data[:label_data.shape[0] // 2]
        elif split_num == 2:
            label_data = label_data[label_data.shape[0] // 2:]

        data = label_data[:, :-1]
        labels = np.clip(label_data[:, -1], 0, 1)
    else:
        data = df.iloc[:, :-1].to_numpy()
        labels = df.iloc[:, -1].clip(0, 1).to_numpy(dtype=np.float32)

    print(sum(labels))
    print("Got dataset of shape " + str(data.shape))
    print("Got label of shape " + str(labels.shape))
    return data, labels
