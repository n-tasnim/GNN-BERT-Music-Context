import torch
import torch.nn as nn

from transformers import DistilBertModel


class TextGenreClassifier(nn.Module):

    def __init__(
        self,
        num_classes=10,
        dropout=0.3,
        freeze_bert=False
    ):
        super().__init__()

        # --------------------------------------------------
        # DistilBERT
        # --------------------------------------------------

        self.bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        # --------------------------------------------------
        # Optionally freeze BERT
        # --------------------------------------------------

        if freeze_bert:

            for parameter in self.bert.parameters():
                parameter.requires_grad = False

        # --------------------------------------------------
        # Classification head
        # --------------------------------------------------

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Linear(
            768,
            num_classes
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        # --------------------------------------------------
        # BERT
        # --------------------------------------------------

        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # DistilBERT has no pooler.
        # First token representation is used.
        text_embedding = output.last_hidden_state[:, 0, :]

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        text_embedding = self.dropout(
            text_embedding
        )

        logits = self.classifier(
            text_embedding
        )

        return logits