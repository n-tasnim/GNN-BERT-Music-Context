import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CNNDataset(Dataset):

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.dataframe = pd.read_csv(csv_path)

        self.label_map = {
            "blues": 0,
            "classical": 1,
            "country": 2,
            "disco": 3,
            "hiphop": 4,
            "jazz": 5,
            "metal": 6,
            "pop": 7,
            "reggae": 8,
            "rock": 9
        }

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        features = np.load(row["feature_path"])

        if features.shape[0] < 6:

            padding = np.zeros(
                (6 - features.shape[0], features.shape[1]),
                dtype=features.dtype
            )

            features = np.concatenate(
                [features, padding],
                axis=0
            )

        elif features.shape[0] > 6:

            features = features[:6]

        features = torch.tensor(
            features,
            dtype=torch.float32
        )

        genre = row["genre"]

        label = self.label_map[genre]

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return features, label