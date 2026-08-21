import torch
from torch_geometric.loader import DataLoader

from src.dataset.graph_dataset import GraphDataset


def calculate_feature_statistics(dataset):

    all_features = []

    print("Calculating feature statistics from training graphs...")

    for i in range(len(dataset)):

        graph = dataset[i]

        all_features.append(
            graph.x
        )

    all_features = torch.cat(
        all_features,
        dim=0
    )

    mean = all_features.mean(
        dim=0
    )

    std = all_features.std(
        dim=0
    )

    
    std[std == 0] = 1.0

    print("Feature mean shape:", mean.shape)
    print("Feature std shape:", std.shape)

    return mean, std


def create_graph_dataloaders(
    train_csv,
    val_csv,
    test_csv,
    batch_size=32
):


    train_dataset = GraphDataset(
        train_csv
    )


    feature_mean, feature_std = calculate_feature_statistics(
        train_dataset
    )

    train_dataset = GraphDataset(
        train_csv,
        feature_mean=feature_mean,
        feature_std=feature_std
    )

    val_dataset = GraphDataset(
        val_csv,
        feature_mean=feature_mean,
        feature_std=feature_std
    )

    test_dataset = GraphDataset(
        test_csv,
        feature_mean=feature_mean,
        feature_std=feature_std
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )