import os

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    average_precision_score
)

from src.dataset.fusion_dataloader import create_fusion_dataloader
from src.models.fusion_model import EarlyConcatFusionModel


# ============================================================
# Configuration
# ============================================================

TEST_CSV = "data/processed/splits/fusion_dataset.csv"

MODEL_PATH = "data/models/best_early_concat.pt"

BATCH_SIZE = 4

NUM_CLASSES = 10

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Load test dataset
# ============================================================

print("\nLoading fusion test dataset...")

test_loader = create_fusion_dataloader(
    TEST_CSV,
    batch_size=BATCH_SIZE,
    shuffle=False,
    split="test"
)

print(
    "Test samples:",
    len(test_loader.dataset)
)


# ============================================================
# Load model
# ============================================================

print("\nLoading trained fusion model...")

model = EarlyConcatFusionModel(
    graph_dim=64,
    text_dim=768,
    hidden_dim=256,
    num_classes=NUM_CLASSES,
    freeze_bert=False
).to(device)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("✓ Fusion model loaded")


# ============================================================
# Evaluation
# ============================================================

all_predictions = []
all_labels = []
all_probabilities = []

total_loss = 0.0
num_batches = 0

criterion = torch.nn.CrossEntropyLoss()


print("\nRunning evaluation...")

with torch.no_grad():

    for batch in test_loader:

        graph = batch["graph"].to(device)

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["labels"].to(device)

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        logits, _, _ = model(
            graph,
            input_ids,
            attention_mask
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            labels
        )

        total_loss += loss.item()

        num_batches += 1

        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = torch.argmax(
            logits,
            dim=1
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


# ============================================================
# Convert to NumPy
# ============================================================

all_predictions = np.array(
    all_predictions
)

all_labels = np.array(
    all_labels
)

all_probabilities = np.array(
    all_probabilities
)


# ============================================================
# Metrics
# ============================================================

test_loss = total_loss / num_batches

accuracy = accuracy_score(
    all_labels,
    all_predictions
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

# ------------------------------------------------------------
# AUC-PR
# ------------------------------------------------------------

# Convert multiclass labels to one-hot representation
one_hot_labels = np.eye(
    NUM_CLASSES
)[all_labels]

macro_auc_pr = average_precision_score(
    one_hot_labels,
    all_probabilities,
    average="macro"
)

micro_auc_pr = average_precision_score(
    one_hot_labels,
    all_probabilities,
    average="micro"
)


# ============================================================
# Final results
# ============================================================

print("\n" + "=" * 60)
print("EARLY-CONCAT TEST RESULTS")
print("=" * 60)

print(
    f"Test Loss:              {test_loss:.4f}"
)

print(
    f"Test Accuracy:          {accuracy:.4f}"
)

print(
    f"Macro Precision:        {macro_precision:.4f}"
)

print(
    f"Macro Recall:           {macro_recall:.4f}"
)

print(
    f"Macro-F1:               {macro_f1:.4f}"
)

print(
    f"Micro-F1:               {micro_f1:.4f}"
)

print(
    f"Macro AUC-PR:           {macro_auc_pr:.4f}"
)

print(
    f"Micro AUC-PR:           {micro_auc_pr:.4f}"
)


# ============================================================
# Per-class report
# ============================================================

print("\n" + "=" * 60)
print("PER-GENRE CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=GENRES,
        zero_division=0
    )
)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(
    "Rows = Actual | Columns = Predicted\n"
)

print(
    "       " +
    " ".join(
        f"{genre[:5]:>6}"
        for genre in GENRES
    )
)

for i, genre in enumerate(GENRES):

    print(
        f"{genre[:5]:>5} " +
        " ".join(
            f"{value:6d}"
            for value in cm[i]
        )
    )


# ============================================================
# Save metrics
# ============================================================

RESULTS_DIR = "data/results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "early_concat_test_results.txt"
)

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write("Early-Concatenation Test Results\n")
    f.write("===================\n\n")

    f.write(
        f"Test Loss: {test_loss:.4f}\n"
    )

    f.write(
        f"Test Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Macro Precision: {macro_precision:.4f}\n"
    )

    f.write(
        f"Macro Recall: {macro_recall:.4f}\n"
    )

    f.write(
        f"Macro-F1: {macro_f1:.4f}\n"
    )

    f.write(
        f"Micro-F1: {micro_f1:.4f}\n"
    )

    f.write(
    f"Macro AUC-PR: {macro_auc_pr:.4f}\n"
)

    f.write(
        f"Micro AUC-PR: {micro_auc_pr:.4f}\n"
    )

    f.write("\n\nClassification Report\n")
    f.write(
        classification_report(
            all_labels,
            all_predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=GENRES,
            zero_division=0
        )
    )

print(
    f"\n✓ Results saved to: {RESULTS_PATH}"
)