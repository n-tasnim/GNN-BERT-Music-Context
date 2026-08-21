import torch
import numpy as np
import pandas as pd

from torch_geometric.loader import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    average_precision_score
)

from src.dataset.graph_dataset import GraphDataset
from src.dataset.dataloader import calculate_feature_statistics
from src.models.graphsage import GraphSAGE


# ============================================================
# Configuration
# ============================================================

TRAIN_CSV = "data/processed/splits/graph_train.csv"

FUSION_CSV = "data/processed/splits/fusion_dataset.csv"

MODEL_PATH = "data/models/best_graphsage.pt"

TEMP_TEST_CSV = "data/processed/splits/graph_task3_test.csv"

BATCH_SIZE = 32

INPUT_DIM = 140
HIDDEN_DIM = 64
NUM_CLASSES = 10


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Calculate ORIGINAL GraphSAGE normalization
# ============================================================

print("\nCalculating GraphSAGE feature statistics...")

train_dataset_for_stats = GraphDataset(
    TRAIN_CSV
)

feature_mean, feature_std = calculate_feature_statistics(
    train_dataset_for_stats
)

print("✓ Using normalization from GraphSAGE training set")


# ============================================================
# Load Task 3 test split
# ============================================================

print("\nLoading Task 3 test split...")

df = pd.read_csv(
    FUSION_CSV
)

test_df = df[
    df["split"] == "test"
].reset_index(drop=True)

print(
    "Task 3 test samples:",
    len(test_df)
)


# ============================================================
# Create temporary CSV
# ============================================================

test_df.to_csv(
    TEMP_TEST_CSV,
    index=False
)

print(
    "Temporary test CSV:",
    TEMP_TEST_CSV
)


# ============================================================
# Dataset
# ============================================================

test_dataset = GraphDataset(
    TEMP_TEST_CSV,
    feature_mean=feature_mean,
    feature_std=feature_std
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Load existing GraphSAGE
# ============================================================

print("\nLoading existing GraphSAGE model...")

model = GraphSAGE(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=NUM_CLASSES
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("✓ GraphSAGE model loaded")
print("✓ No retraining performed")


# ============================================================
# Evaluation
# ============================================================

all_predictions = []
all_labels = []
all_probabilities = []


print("\nRunning GNN-only Task 3 evaluation...")

with torch.no_grad():

    for batch in test_loader:

        batch = batch.to(device)

        logits = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            batch.y.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


# ============================================================
# NumPy
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

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


# One-hot encoding for multiclass AUC-PR
one_hot_labels = np.eye(
    NUM_CLASSES
)[all_labels]


macro_auc_pr = average_precision_score(
    one_hot_labels,
    all_probabilities,
    average="macro"
)


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 60)
print("TASK 3 GNN-ONLY ABLATION")
print("=" * 60)

print(
    f"Test Samples:     {len(all_labels)}"
)

print(
    f"Test Accuracy:    {accuracy:.4f}"
)

print(
    f"Macro-F1:         {macro_f1:.4f}"
)

print(
    f"Macro AUC-PR:     {macro_auc_pr:.4f}"
)


# ============================================================
# Save
# ============================================================

RESULTS_PATH = (
    "data/results/graphsage_task3_ablation.txt"
)

with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Task 3 GNN-Only Ablation\n"
    )
    f.write(
        "========================\n\n"
    )

    f.write(
        f"Test Samples: {len(all_labels)}\n"
    )

    f.write(
        f"Test Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Macro-F1: {macro_f1:.4f}\n"
    )

    f.write(
        f"Macro AUC-PR: {macro_auc_pr:.4f}\n"
    )


print(
    f"\n✓ Results saved to: {RESULTS_PATH}"
)