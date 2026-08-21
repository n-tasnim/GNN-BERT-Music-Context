import ast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BERTMusicDataset(Dataset):

    def __init__(self, csv_path, top_tags):
        self.dataframe = pd.read_csv(csv_path)
        self.top_tags = top_tags

        # Map each tag to an index
        self.tag_to_index = {
            tag: index
            for index, tag in enumerate(top_tags)
        }

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        # Caption
        caption = str(row["caption"])

        # aspect_list is stored as a string representation
        # of a Python list in the CSV
        try:
            tags = ast.literal_eval(row["aspect_list"])
        except (ValueError, SyntaxError):
            tags = []

        # Multi-label vector
        labels = np.zeros(
            len(self.top_tags),
            dtype=np.float32
        )

        for tag in tags:

            if tag in self.tag_to_index:

                tag_index = self.tag_to_index[tag]

                labels[tag_index] = 1.0

        labels = torch.tensor(
            labels,
            dtype=torch.float32
        )

        return caption, labels