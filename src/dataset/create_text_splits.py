import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.config import (
    MUSICCAPS_CSV,
    SPLIT_OUTPUT_PATH,
    TEXT_TRAIN_CSV,
    TEXT_VAL_CSV,
    TEXT_TEST_CSV,
)

RANDOM_STATE = 42


def load_musiccaps():
    return pd.read_csv(MUSICCAPS_CSV)


def split_dataset(df):

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    return train_df, val_df, test_df


def verify(train_df, val_df, test_df):

    train = set(train_df["ytid"])
    val = set(val_df["ytid"])
    test = set(test_df["ytid"])

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)

    print("No data leakage detected.")


def save(train_df, val_df, test_df):

    SPLIT_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TEXT_TRAIN_CSV, index=False)
    val_df.to_csv(TEXT_VAL_CSV, index=False)
    test_df.to_csv(TEXT_TEST_CSV, index=False)


def main():

    print("Loading MusicCaps...")

    df = load_musiccaps()

    print(f"Total captions: {len(df)}")

    train_df, val_df, test_df = split_dataset(df)

    verify(train_df, val_df, test_df)

    save(train_df, val_df, test_df)

    print()

    print(f"Train      : {len(train_df)}")
    print(f"Validation : {len(val_df)}")
    print(f"Test       : {len(test_df)}")

    print("\nText splits saved successfully.")


if __name__ == "__main__":
    main()