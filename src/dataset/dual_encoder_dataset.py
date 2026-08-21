import pickle

import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch_geometric.data import Data, Batch


class DualEncoderDataset(Dataset):

    def __init__(
        self,
        csv_path,
        split=None,
        max_length=256
    ):

        self.df = pd.read_csv(csv_path)

        # ----------------------------------------------------
        # Select split
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

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

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
        # Graph path
        # ----------------------------------------------------

        graph_path = row[
            "graph_path"
        ]

        return {
            "graph_path": graph_path,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "caption": caption,
            "genre": row["genre"]
        }


def load_graph(graph_path):

    # --------------------------------------------------------
    # Load NetworkX graph
    # --------------------------------------------------------

    with open(
        graph_path,
        "rb"
    ) as f:

        nx_graph = pickle.load(f)

    # --------------------------------------------------------
    # Extract node features
    # --------------------------------------------------------

    node_features = []

    for node in nx_graph.nodes():

        features = nx_graph.nodes[node].get(
            "features"
        )

        if features is None:

            features = nx_graph.nodes[node].get(
                "x"
            )

        if features is None:

            raise ValueError(
                f"No node features found for node "
                f"{node} in {graph_path}"
            )

        node_features.append(
            features
        )

    node_features = torch.tensor(
        node_features,
        dtype=torch.float
    )

    # --------------------------------------------------------
    # Create node index mapping
    # --------------------------------------------------------

    node_list = list(
        nx_graph.nodes()
    )

    node_to_idx = {
        node: i
        for i, node in enumerate(node_list)
    }

    # --------------------------------------------------------
    # Create bidirectional edges
    # --------------------------------------------------------

    edges = []

    for source, target in nx_graph.edges():

        edges.append([
            node_to_idx[source],
            node_to_idx[target]
        ])

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

    # --------------------------------------------------------
    # Create PyG graph
    # --------------------------------------------------------

    graph = Data(
        x=node_features,
        edge_index=edge_index
    )

    return graph


def collate_dual_encoder(batch):

    # --------------------------------------------------------
    # Load graphs
    # --------------------------------------------------------

    graphs = []

    for item in batch:

        graph = load_graph(
            item["graph_path"]
        )

        graphs.append(
            graph
        )

    graph_batch = Batch.from_data_list(
        graphs
    )

    # --------------------------------------------------------
    # Stack text tensors
    # --------------------------------------------------------

    input_ids = torch.stack([
        item["input_ids"]
        for item in batch
    ])

    attention_masks = torch.stack([
        item["attention_mask"]
        for item in batch
    ])

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    captions = [
        item["caption"]
        for item in batch
    ]

    genres = [
        item["genre"]
        for item in batch
    ]

    return {
        "graph": graph_batch,
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "captions": captions,
        "genres": genres
    }