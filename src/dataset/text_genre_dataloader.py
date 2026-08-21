from torch.utils.data import DataLoader

from src.dataset.text_genre_dataset import TextGenreDataset


def create_text_genre_dataloader(
    csv_path,
    batch_size=8,
    shuffle=False,
    split=None
):

    dataset = TextGenreDataset(
        csv_path,
        split=split
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader