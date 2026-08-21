from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np
from pathlib import Path

from src.text.text_loader import load_musiccaps
from src.preprocessing.config import TOKEN_OUTPUT_PATH

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_caption(caption):

    encoding = tokenizer(
        caption,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np"
    )

    return encoding

def save_tokens(encoding, index):

    TOKEN_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    np.savez(
        TOKEN_OUTPUT_PATH / f"{index}.npz",
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"]
    )

def process_dataset():

    df = load_musiccaps()

    for index, row in tqdm(df.iterrows(), total=len(df)):

        caption = row["caption"]

        encoding = tokenize_caption(caption)

        save_tokens(encoding, index)

    print(f"\nProcessed {len(df)} captions.")

if __name__ == "__main__":
    process_dataset()