import pickle
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.utils import from_networkx


class GraphDataset(Dataset):

    def __init__(self, csv_path, feature_mean=None, feature_std=None):

        self.csv_path = Path(csv_path)
        self.dataframe = pd.read_csv(self.csv_path)

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

        
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        graph_path = row["graph_path"]

        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        data = from_networkx(
            graph,
            group_node_attrs=["features"]
        )

        data.x = data.x.float()


        if self.feature_mean is not None and self.feature_std is not None:

            data.x = (
                data.x - self.feature_mean
            ) / self.feature_std

        genre = row["genre"]

        label = self.label_map[genre]

        data.y = torch.tensor(
            [label],
            dtype=torch.long
        )

        data.genre = genre
        data.graph_path = graph_path

        return data