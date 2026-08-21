import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, global_mean_pool
from transformers import DistilBertModel


class GraphEncoder(nn.Module):

    def __init__(
        self,
        input_dim=140,
        hidden_dim=64,
        dropout=0.3
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

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        edge_index,
        batch
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        # One embedding per graph
        graph_embedding = global_mean_pool(
            x,
            batch
        )

        return graph_embedding


class TextEncoder(nn.Module):

    def __init__(
        self,
        freeze_bert=False
    ):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        if freeze_bert:

            for parameter in self.bert.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # DistilBERT does not have a CLS pooler.
        # First token representation acts as CLS representation.
        text_embedding = output.last_hidden_state[:, 0, :]

        return text_embedding


class EarlyConcatFusion(nn.Module):

    def __init__(
        self,
        graph_dim=64,
        text_dim=768,
        hidden_dim=256,
        num_classes=10
    ):
        super().__init__()

        self.classifier = nn.Sequential(

            nn.Linear(
                graph_dim + text_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                hidden_dim,
                num_classes
            )
        )

    def forward(
        self,
        graph_embedding,
        text_embedding
    ):

        z = torch.cat(
            [
                graph_embedding,
                text_embedding
            ],
            dim=1
        )

        return self.classifier(z)


class CrossAttentionFusion(nn.Module):

    def __init__(
        self,
        graph_dim=64,
        text_dim=768,
        attention_dim=128,
        hidden_dim=256,
        num_classes=10
    ):
        super().__init__()

        # Graph embedding → Query
        self.query = nn.Linear(
            graph_dim,
            attention_dim
        )

        # BERT token embeddings → Keys / Values
        self.key = nn.Linear(
            text_dim,
            attention_dim
        )

        self.value = nn.Linear(
            text_dim,
            attention_dim
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                graph_dim + attention_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                hidden_dim,
                num_classes
            )
        )

    def forward(
        self,
        graph_embedding,
        text_hidden_states,
        attention_mask
    ):

        # --------------------------------------------------
        # Q = graph embedding
        # --------------------------------------------------

        q = self.query(
            graph_embedding
        )

        # Shape:
        # [batch, attention_dim]
        q = q.unsqueeze(1)

        # --------------------------------------------------
        # K and V = BERT token representations
        # --------------------------------------------------

        k = self.key(
            text_hidden_states
        )

        v = self.value(
            text_hidden_states
        )

        # --------------------------------------------------
        # Attention scores
        # --------------------------------------------------

        scores = torch.matmul(
            q,
            k.transpose(
                1,
                2
            )
        )

        scores = scores / (
            self.query.out_features ** 0.5
        )

        # --------------------------------------------------
        # Mask padding tokens
        # --------------------------------------------------

        mask = attention_mask.unsqueeze(1)

        scores = scores.masked_fill(
            mask == 0,
            -1e9
        )

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        # --------------------------------------------------
        # A H_text
        # --------------------------------------------------

        attended_text = torch.matmul(
            attention_weights,
            v
        )

        attended_text = attended_text.squeeze(1)

        # --------------------------------------------------
        # z = CONCAT(g, AH_text)
        # --------------------------------------------------

        z = torch.cat(
            [
                graph_embedding,
                attended_text
            ],
            dim=1
        )

        logits = self.classifier(
            z
        )

        return logits, attention_weights, z


class FusionModel(nn.Module):

    def __init__(
        self,
        graph_dim=64,
        text_dim=768,
        attention_dim=128,
        hidden_dim=256,
        num_classes=10,
        freeze_bert=False
    ):
        super().__init__()

        self.graph_encoder = GraphEncoder(
            input_dim=140,
            hidden_dim=graph_dim
        )

        self.text_encoder = TextEncoder(
            freeze_bert=freeze_bert
        )

        self.fusion = CrossAttentionFusion(
            graph_dim=graph_dim,
            text_dim=text_dim,
            attention_dim=attention_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

    def forward(
        self,
        graph,
        input_ids,
        attention_mask
    ):

        # --------------------------------------------------
        # Graph branch
        # --------------------------------------------------

        graph_embedding = self.graph_encoder(
            graph.x,
            graph.edge_index,
            graph.batch
        )

        # --------------------------------------------------
        # BERT branch
        # --------------------------------------------------

        bert_output = self.text_encoder.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_hidden_states = (
            bert_output.last_hidden_state
        )

        # --------------------------------------------------
        # Cross-attention fusion
        # --------------------------------------------------

        logits, attention_weights, z = self.fusion(
            graph_embedding,
            text_hidden_states,
            attention_mask
        )

        return (
            logits,
            graph_embedding,
            text_hidden_states,
            attention_weights,
            z
        )

class EarlyConcatFusionModel(nn.Module):

    def __init__(
        self,
        graph_dim=64,
        text_dim=768,
        hidden_dim=256,
        num_classes=10,
        freeze_bert=False
    ):
        super().__init__()

        # -----------------------------
        # Graph encoder
        # -----------------------------

        self.graph_encoder = GraphEncoder(
            input_dim=140,
            hidden_dim=graph_dim
        )

        # -----------------------------
        # Text encoder
        # -----------------------------

        self.text_encoder = TextEncoder(
            freeze_bert=freeze_bert
        )

        # -----------------------------
        # Early concatenation
        # -----------------------------

        self.fusion = EarlyConcatFusion(
            graph_dim=graph_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

    def forward(
        self,
        graph,
        input_ids,
        attention_mask
    ):

        # -----------------------------
        # Graph branch
        # -----------------------------

        graph_embedding = self.graph_encoder(
            graph.x,
            graph.edge_index,
            graph.batch
        )

        # -----------------------------
        # BERT branch
        # -----------------------------

        text_embedding = self.text_encoder(
            input_ids,
            attention_mask
        )

        # -----------------------------
        # Concatenate
        # -----------------------------

        logits = self.fusion(
            graph_embedding,
            text_embedding
        )

        return (
            logits,
            graph_embedding,
            text_embedding
        )