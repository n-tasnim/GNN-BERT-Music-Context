import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextGenreDataset(Dataset):

    def __init__(
        self,
        csv_path,
        split=None,
        max_length=256
    ):
        self.df = pd.read_csv(csv_path)

        # ----------------------------------------------------
        # Select train / validation / test split
        # ----------------------------------------------------

        if split is not None:

            self.df = self.df[
                self.df["split"] == split
            ].reset_index(drop=True)

        # ----------------------------------------------------
        # Tokenizer
        # ----------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )

        self.max_length = max_length

        # ----------------------------------------------------
        # Genre labels
        # ----------------------------------------------------

        self.genre_to_idx = {
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

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # ----------------------------------------------------
        # Caption
        # ----------------------------------------------------

        caption = str(
            row["caption"]
        )

        encoded = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoded[
            "input_ids"
        ].squeeze(0)

        attention_mask = encoded[
            "attention_mask"
        ].squeeze(0)

        # ----------------------------------------------------
        # Genre label
        # ----------------------------------------------------

        genre = row["genre"]

        label = self.genre_to_idx[
            genre
        ]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(
                label,
                dtype=torch.long
            ),
            "caption": caption,
            "genre": genre
        }