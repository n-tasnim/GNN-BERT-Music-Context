from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.preprocessing.config import (
    GRAPH_OUTPUT_PATH,
    FEATURE_OUTPUT_PATH,
    SPLIT_OUTPUT_PATH,
    GRAPH_TRAIN_CSV,
    GRAPH_VAL_CSV,
    GRAPH_TEST_CSV,
)


RANDOM_STATE = 42


def collect_graph_files():

    data = []

    genres = sorted([folder for folder in GRAPH_OUTPUT_PATH.iterdir() if folder.is_dir()])

    for genre_folder in genres:

        genre = genre_folder.name

        graph_files = sorted(genre_folder.glob("*.pkl"))

        for graph_file in tqdm(graph_files, desc=f"Scanning {genre}"):

            feature_file = (
                FEATURE_OUTPUT_PATH
                / genre
                / graph_file.with_suffix(".npy").name
            )

            data.append(
                {
                    "genre": genre,
                    "graph_path": str(graph_file),
                    "feature_path": str(feature_file),
                }
            )

    df = pd.DataFrame(data)

    return df


def split_dataset(df):

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["genre"],
        random_state=RANDOM_STATE,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["genre"],
        random_state=RANDOM_STATE,
    )

    return train_df, val_df, test_df


def verify_no_overlap(train_df, val_df, test_df):

    train_set = set(train_df["graph_path"])
    val_set = set(val_df["graph_path"])
    test_set = set(test_df["graph_path"])

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    print("No data leakage detected.")


def save_splits(train_df, val_df, test_df):

    SPLIT_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(GRAPH_TRAIN_CSV, index=False)
    val_df.to_csv(GRAPH_VAL_CSV, index=False)
    test_df.to_csv(GRAPH_TEST_CSV, index=False)


def print_statistics(train_df, val_df, test_df):

    print("\n==============================")
    print("Dataset Split Statistics")
    print("==============================")

    print(f"Training   : {len(train_df)}")
    print(f"Validation : {len(val_df)}")
    print(f"Testing    : {len(test_df)}")

    print("\nTraining Distribution")

    print(train_df["genre"].value_counts().sort_index())

    print("\nValidation Distribution")

    print(val_df["genre"].value_counts().sort_index())

    print("\nTesting Distribution")

    print(test_df["genre"].value_counts().sort_index())


def main():

    print("Collecting graph files...")

    df = collect_graph_files()

    print(f"\nTotal graphs found: {len(df)}")

    train_df, val_df, test_df = split_dataset(df)

    verify_no_overlap(train_df, val_df, test_df)

    save_splits(train_df, val_df, test_df)

    print_statistics(train_df, val_df, test_df)

    print("\nGraph splits saved successfully.")


if __name__ == "__main__":
    main()