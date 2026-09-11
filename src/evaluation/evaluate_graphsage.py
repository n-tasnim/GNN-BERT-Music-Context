import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.dataset.dataloader import create_graph_dataloaders
from src.models.graphsage import GraphSAGE


# ============================================================
# Configuration
# ============================================================

TEST_CSV = "data/processed/splits/graph_test.csv"
MODEL_PATH = "data/models/best_graphsage.pt"

BATCH_SIZE = 32

INPUT_DIM = 140
HIDDEN_DIM = 64
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
# DataLoader
# ============================================================

# We only need the test loader here.
_, _, test_loader = create_graph_dataloaders(
    TEST_CSV,
    TEST_CSV,
    TEST_CSV,
    batch_size=BATCH_SIZE
)

print("Test graphs:", len(test_loader.dataset))


# ============================================================
# Model
# ============================================================

model = GraphSAGE(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=NUM_CLASSES
).to(device)


# ============================================================
# Load trained model
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# Collect predictions
# ============================================================

all_predictions = []
all_labels = []
all_probabilities = []


with torch.no_grad():

    for batch in test_loader:

        batch = batch.to(device)

        output = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predictions = output.argmax(
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


all_predictions = np.array(
    all_predictions
)

all_labels = np.array(
    all_labels
)


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


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

print("\n==============================")
print("GraphSAGE Evaluation")
print("==============================")

print(
    f"Accuracy:         {accuracy:.4f}"
)

print(
    f"Macro Precision:  {precision:.4f}"
)

print(
    f"Macro Recall:     {recall:.4f}"
)

print(
    f"Macro F1:         {f1:.4f}"
)

print(
    f"Macro AUC-PR:    {macro_auc_pr:.4f}"
)


print("\n==============================")
print("Per-Genre Classification Report")
print("==============================")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=GENRES,
        zero_division=0
    )
)


cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n==============================")
print("Confusion Matrix")
print("==============================")

print(cm)