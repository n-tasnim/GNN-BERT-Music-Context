import pickle

import numpy as np
import torch

from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch

from src.dataset.fusion_pytorch_dataset import FusionDataset


def load_graph(graph_path):

    # --------------------------------------------------
    # Load NetworkX graph
    # --------------------------------------------------

    with open(graph_path, "rb") as f:
        nx_graph = pickle.load(f)

    # --------------------------------------------------
    # Extract node features
    # --------------------------------------------------

    node_features = []

    for node in nx_graph.nodes():

        features = nx_graph.nodes[node].get("x")

        if features is None:
            features = nx_graph.nodes[node].get(
                "features"
            )

        if features is None:
            raise ValueError(
                f"No node features found for node {node} "
                f"in graph:\n{graph_path}"
            )

        node_features.append(features)

    # --------------------------------------------------
    # Convert features to NumPy array
    # --------------------------------------------------

    node_features = np.asarray(
        node_features,
        dtype=np.float32
    )

    # --------------------------------------------------
    # Create edge index
    # --------------------------------------------------

    node_list = list(nx_graph.nodes())

    node_to_idx = {
        node: i
        for i, node in enumerate(node_list)
    }

    edges = []

    for source, target in nx_graph.edges():

        edges.append([
            node_to_idx[source],
            node_to_idx[target]
        ])

        # Add reverse edge
        edges.append([
            node_to_idx[target],
            node_to_idx[source]
        ])

    if len(edges) > 0:

        edge_index = torch.tensor(
            edges,
            dtype=torch.long
        ).t().contiguous()

    else:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

    # --------------------------------------------------
    # Create PyG graph
    # --------------------------------------------------

    data = Data(
        x=torch.from_numpy(node_features),
        edge_index=edge_index
    )

    return data


def collate_fusion(batch):

    graphs = []

    input_ids = []
    attention_masks = []
    labels = []

    captions = []
    genres = []

    for item in batch:

        # --------------------------------------------------
        # Load graph
        # --------------------------------------------------

        graph = load_graph(
            item["graph_path"]
        )

        graphs.append(graph)

        # --------------------------------------------------
        # Text
        # --------------------------------------------------

        input_ids.append(
            item["input_ids"]
        )

        attention_masks.append(
            item["attention_mask"]
        )

        # --------------------------------------------------
        # Labels
        # --------------------------------------------------

        labels.append(
            item["label"]
        )

        captions.append(
            item["caption"]
        )

        genres.append(
            item["genre"]
        )

    # ------------------------------------------------------
    # Batch graphs
    # ------------------------------------------------------

    graph_batch = Batch.from_data_list(
        graphs
    )

    # ------------------------------------------------------
    # Batch text
    # ------------------------------------------------------

    input_ids = torch.stack(
        input_ids
    )

    attention_masks = torch.stack(
        attention_masks
    )

    labels = torch.stack(
        labels
    )

    return {
        "graph": graph_batch,
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
        "captions": captions,
        "genres": genres
    }


def create_fusion_dataloader(
    csv_path,
    batch_size=8,
    shuffle=False,
    split=None
):

    dataset = FusionDataset(
        csv_path,
        split=split
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fusion
    )

    return loader


def create_fusion_dataloaders(
    train_csv,
    val_csv,
    test_csv,
    batch_size=8
):

    train_loader = create_fusion_dataloader(
        train_csv,
        batch_size=batch_size,
        shuffle=True,
        split="train"
    )

    val_loader = create_fusion_dataloader(
        val_csv,
        batch_size=batch_size,
        shuffle=False,
        split="val"
    )

    test_loader = create_fusion_dataloader(
        test_csv,
        batch_size=batch_size,
        shuffle=False,
        split="test"
    )

    return train_loader, val_loader, test_loader