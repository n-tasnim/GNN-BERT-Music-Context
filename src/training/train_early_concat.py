import os

import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.dataset.fusion_dataloader import (
    create_fusion_dataloader
)

from src.models.fusion_model import (
    EarlyConcatFusionModel
)


# ============================================================
# Configuration
# ============================================================

TRAIN_CSV = "data/processed/splits/fusion_dataset.csv"
VAL_CSV = "data/processed/splits/fusion_dataset.csv"
TEST_CSV = "data/processed/splits/fusion_dataset.csv"

MODEL_DIR = "data/models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_early_concat.pt"
)

BATCH_SIZE = 8
LEARNING_RATE = 0.0001
EPOCHS = 20

NUM_CLASSES = 10


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Dataloaders
# ============================================================

print("\nLoading early-concat datasets...")

train_loader = create_fusion_dataloader(
    TRAIN_CSV,
    batch_size=BATCH_SIZE,
    shuffle=True,
    split="train"
)

val_loader = create_fusion_dataloader(
    VAL_CSV,
    batch_size=BATCH_SIZE,
    shuffle=False,
    split="val"
)

test_loader = create_fusion_dataloader(
    TEST_CSV,
    batch_size=BATCH_SIZE,
    shuffle=False,
    split="test"
)

print(
    "Training samples:",
    len(train_loader.dataset)
)

print(
    "Validation samples:",
    len(val_loader.dataset)
)

print(
    "Test samples:",
    len(test_loader.dataset)
)


# ============================================================
# Model
# ============================================================

print("\nLoading early-concat model...")

model = EarlyConcatFusionModel(
    graph_dim=64,
    text_dim=768,
    hidden_dim=256,
    num_classes=NUM_CLASSES,
    freeze_bert=False
).to(device)


# ============================================================
# Optimizer
# ============================================================

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Train
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:

        graph = batch["graph"].to(device)

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        optimizer.zero_grad()

        logits, _, _ = model(
            graph,
            input_ids,
            attention_mask
        )

        loss = F.cross_entropy(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    average_loss = (
        total_loss /
        len(train_loader)
    )

    accuracy = (
        correct /
        total
    )

    return average_loss, accuracy


# ============================================================
# Evaluation
# ============================================================

def evaluate(loader):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch in loader:

            graph = batch["graph"].to(device)

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            logits, _, _ = model(
                graph,
                input_ids,
                attention_mask
            )

            loss = F.cross_entropy(
                logits,
                labels
            )

            total_loss += loss.item()

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = (
        total_loss /
        len(loader)
    )

    accuracy = (
        correct /
        total
    )

    return average_loss, accuracy


# ============================================================
# Training loop
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

best_val_accuracy = 0.0

print("\n==============================")
print("Starting Early-Concat Training")
print("==============================")


for epoch in range(
    1,
    EPOCHS + 1
):

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    val_loss, val_accuracy = (
        evaluate(val_loader)
    )

    print(
        f"\nEpoch {epoch:02d}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Acc: {train_accuracy:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Acc: {val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"✓ Best early-concat model saved "
            f"(Val Acc: {val_accuracy:.4f})"
        )


# ============================================================
# Final test evaluation
# ============================================================

print(
    "\nLoading best early-concat model..."
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

test_loss, test_accuracy = evaluate(
    test_loader
)

print(
    "\n=============================="
)

print(
    "Early-Concat Final Results"
)

print(
    "=============================="
)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.4f}"
)