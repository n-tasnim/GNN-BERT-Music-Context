import pandas as pd

from src.preprocessing.config import RAW_TEXT_PATH


def load_musiccaps():

    csv_path = RAW_TEXT_PATH / "musiccaps-public.csv"

    df = pd.read_csv(csv_path)

    return df

if __name__ == "__main__":

    df = load_musiccaps()

    print(df.head())

    print()

    print(df.columns)