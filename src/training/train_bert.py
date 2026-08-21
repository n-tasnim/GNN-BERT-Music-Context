import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)

from src.dataset.bert_dataset import BERTMusicDataset
from src.dataset.top_tags import TOP_50_TAGS
from src.models.bert_classifier import BERTMusicClassifier
from src.training.bert_utils import tokenize_captions


# ============================================================
# Configuration
# ============================================================

TRAIN_CSV = "data/processed/splits/text_train.csv"
VAL_CSV = "data/processed/splits/text_val.csv"
TEST_CSV = "data/processed/splits/text_test.csv"

MODEL_DIR = "data/models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_bert.pt"
)

NUM_LABELS = 50
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5
MAX_LENGTH = 256


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Dataset
# ============================================================

train_dataset = BERTMusicDataset(
    TRAIN_CSV,
    TOP_50_TAGS
)

val_dataset = BERTMusicDataset(
    VAL_CSV,
    TOP_50_TAGS
)

test_dataset = BERTMusicDataset(
    TEST_CSV,
    TOP_50_TAGS
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


# ============================================================
# Collate function
# ============================================================

def collate_fn(batch):

    captions = [
        item[0]
        for item in batch
    ]

    labels = torch.stack([
        item[1]
        for item in batch
    ])

    encoded = tokenize_captions(
        captions,
        max_length=MAX_LENGTH
    )

    return (
        encoded["input_ids"],
        encoded["attention_mask"],
        labels
    )


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


# ============================================================
# Test first batch
# ============================================================

input_ids, attention_mask, labels = next(
    iter(train_loader)
)

print("\nFirst batch:")
print("Input IDs:", input_ids.shape)
print("Attention mask:", attention_mask.shape)
print("Labels:", labels.shape)


# ============================================================
# Model
# ============================================================

model = BERTMusicClassifier(
    num_labels=NUM_LABELS
).to(device)


# ============================================================
# Loss and optimizer
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Evaluation function
# ============================================================

def evaluate(loader):

    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0

    with torch.no_grad():

        for input_ids, attention_mask, labels in loader:

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(
                logits,
                labels
            )

            total_loss += loss.item()

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= 0.5
            ).float()

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_labels.append(
                labels.cpu().numpy()
            )

    all_predictions = np.concatenate(
        all_predictions,
        axis=0
    )

    all_labels = np.concatenate(
        all_labels,
        axis=0
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    micro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="micro",
        zero_division=0
    )

    macro_precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    average_loss = (
        total_loss / len(loader)
    )

    return (
        average_loss,
        macro_f1,
        micro_f1,
        macro_precision,
        macro_recall
    )


# ============================================================
# Training
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 25 == 0:
            print(
                f"  Batch {batch_idx + 1}/{len(train_loader)} "
                f"| Loss: {loss.item():.4f}"
            )

    return total_loss / len(train_loader)


# ============================================================
# Training loop
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

best_val_f1 = 0.0

history = []

for epoch in range(1, EPOCHS + 1):

    train_loss = train_one_epoch()

    (
        val_loss,
        val_macro_f1,
        val_micro_f1,
        val_precision,
        val_recall
    ) = evaluate(val_loader)

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "macro_f1": val_macro_f1,
        "micro_f1": val_micro_f1
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Macro-F1: {val_macro_f1:.4f} | "
        f"Micro-F1: {val_micro_f1:.4f}"
    )

    if val_macro_f1 > best_val_f1:

        best_val_f1 = val_macro_f1

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"  ✓ Best model saved "
            f"(Val Macro-F1: {val_macro_f1:.4f})"
        )


# ============================================================
# Final test evaluation
# ============================================================

print("\nLoading best BERT model...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

(
    test_loss,
    test_macro_f1,
    test_micro_f1,
    test_precision,
    test_recall
) = evaluate(test_loader)


print("\n==============================")
print("BERT Final Results")
print("==============================")

print(
    f"Best Validation Macro-F1: "
    f"{best_val_f1:.4f}"
)

print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test Macro-F1: "
    f"{test_macro_f1:.4f}"
)

print(
    f"Test Micro-F1: "
    f"{test_micro_f1:.4f}"
)

print(
    f"Test Macro Precision: "
    f"{test_precision:.4f}"
)

print(
    f"Test Macro Recall: "
    f"{test_recall:.4f}"
)

print(
    f"\nBest model saved to: "
    f"{MODEL_PATH}"
)