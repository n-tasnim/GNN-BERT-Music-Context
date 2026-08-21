import torch
import torch.nn as nn
from transformers import AutoModel


class BERTMusicClassifier(nn.Module):

    def __init__(
        self,
        num_labels=50,
        model_name="distilbert-base-uncased"
    ):
        super().__init__()

        # Pretrained DistilBERT
        self.bert = AutoModel.from_pretrained(
            model_name
        )

        # DistilBERT hidden size
        hidden_size = self.bert.config.hidden_size

        # Classification layer
        self.classifier = nn.Linear(
            hidden_size,
            num_labels
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # DistilBERT does not have a pooler/[CLS] output.
        # We use the representation of the first token.
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(
            cls_embedding
        )

        return logits