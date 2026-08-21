import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.dataset.cnn_dataset import CNNDataset
from src.models.cnn import AudioCNN


# ============================================================
# Configuration
# ============================================================

TEST_CSV = "data/processed/splits/graph_test.csv"
MODEL_PATH = "data/models/best_cnn.pt"

BATCH_SIZE = 32
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
# Test Dataset
# ============================================================

test_dataset = CNNDataset(
    TEST_CSV
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test samples:", len(test_dataset))


# ============================================================
# Model
# ============================================================

model = AudioCNN(
    num_classes=NUM_CLASSES
).to(device)

print("\nLoading best CNN model...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# Evaluation
# ============================================================

all_predictions = []
all_labels = []

with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(device)
        labels = labels.to(device)

        features = features[:, :, :128]

        outputs = model(features)

        predictions = outputs.argmax(
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ============================================================
# Metrics
# ============================================================

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


# ============================================================
# Results
# ============================================================

print("\n==============================")
print("CNN Evaluation Results")
print("==============================")

print(
    f"Accuracy:         {accuracy:.4f}"
)

print(
    f"Macro Precision:  {macro_precision:.4f}"
)

print(
    f"Macro Recall:     {macro_recall:.4f}"
)

print(
    f"Macro F1:         {macro_f1:.4f}"
)


# ============================================================
# Classification Report
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=GENRES,
        zero_division=0
    )
)


# ============================================================
# Confusion Matrix
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print(cm)