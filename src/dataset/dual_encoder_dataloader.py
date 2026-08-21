from torch.utils.data import DataLoader

from src.dataset.dual_encoder_dataset import (
    DualEncoderDataset,
    collate_dual_encoder
)


def create_dual_encoder_dataloader(
    csv_path,
    batch_size=8,
    shuffle=False,
    split=None
):

    dataset = DualEncoderDataset(
        csv_path=csv_path,
        split=split,
        max_length=256
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_dual_encoder
    )

    return loader


def create_dual_encoder_dataloaders(
    csv_path,
    batch_size=8
):

    train_loader = create_dual_encoder_dataloader(
        csv_path,
        batch_size=batch_size,
        shuffle=True,
        split="train"
    )

    val_loader = create_dual_encoder_dataloader(
        csv_path,
        batch_size=batch_size,
        shuffle=False,
        split="val"
    )

    test_loader = create_dual_encoder_dataloader(
        csv_path,
        batch_size=batch_size,
        shuffle=False,
        split="test"
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )