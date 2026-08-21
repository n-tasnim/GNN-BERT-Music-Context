import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, global_mean_pool
from transformers import AutoModel


class GraphEncoder(nn.Module):

    def __init__(
        self,
        input_dim=140,
        hidden_dim=128,
        embedding_dim=256
    ):
        super().__init__()

        self.conv1 = SAGEConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = SAGEConv(
            hidden_dim,
            hidden_dim
        )

        self.projection = nn.Linear(
            hidden_dim,
            embedding_dim
        )

    def forward(
        self,
        x,
        edge_index,
        batch
    ):

        # ----------------------------------------------------
        # GraphSAGE
        # ----------------------------------------------------

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        # ----------------------------------------------------
        # Graph-level representation
        # ----------------------------------------------------

        x = global_mean_pool(
            x,
            batch
        )

        # ----------------------------------------------------
        # Project into shared embedding space
        # ----------------------------------------------------

        x = self.projection(
            x
        )

        # ----------------------------------------------------
        # L2 normalization
        # ----------------------------------------------------

        x = F.normalize(
            x,
            p=2,
            dim=1
        )

        return x


class TextEncoder(nn.Module):

    def __init__(
        self,
        embedding_dim=256,
        freeze_bert=True
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(
            "distilbert-base-uncased"
        )

        if freeze_bert:

            for parameter in self.bert.parameters():
                parameter.requires_grad = False

        self.projection = nn.Linear(
            768,
            embedding_dim
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        # ----------------------------------------------------
        # DistilBERT
        # ----------------------------------------------------

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # ----------------------------------------------------
        # CLS representation
        # ----------------------------------------------------

        text_embedding = outputs.last_hidden_state[:, 0, :]

        # ----------------------------------------------------
        # Project into shared embedding space
        # ----------------------------------------------------

        text_embedding = self.projection(
            text_embedding
        )

        # ----------------------------------------------------
        # L2 normalization
        # ----------------------------------------------------

        text_embedding = F.normalize(
            text_embedding,
            p=2,
            dim=1
        )

        return text_embedding


class DualEncoder(nn.Module):

    def __init__(
        self,
        graph_input_dim=140,
        graph_hidden_dim=128,
        embedding_dim=256,
        freeze_bert=True
    ):
        super().__init__()

        self.graph_encoder = GraphEncoder(
            input_dim=graph_input_dim,
            hidden_dim=graph_hidden_dim,
            embedding_dim=embedding_dim
        )

        self.text_encoder = TextEncoder(
            embedding_dim=embedding_dim,
            freeze_bert=freeze_bert
        )

    def forward(
        self,
        graph,
        input_ids,
        attention_mask
    ):

        # ----------------------------------------------------
        # Audio graph embedding
        # ----------------------------------------------------

        graph_embedding = self.graph_encoder(
            x=graph.x,
            edge_index=graph.edge_index,
            batch=graph.batch
        )

        # ----------------------------------------------------
        # Text embedding
        # ----------------------------------------------------

        text_embedding = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        return (
            graph_embedding,
            text_embedding
        )