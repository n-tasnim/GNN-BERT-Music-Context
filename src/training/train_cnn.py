import os

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.dataset.cnn_dataset import CNNDataset
from src.models.cnn import AudioCNN


TRAIN_CSV = "data/processed/splits/graph_train.csv"
VAL_CSV = "data/processed/splits/graph_val.csv"
TEST_CSV = "data/processed/splits/graph_test.csv"

MODEL_DIR = "data/models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_cnn.pt"
)

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30

NUM_CLASSES = 10

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


train_dataset = CNNDataset(TRAIN_CSV)
val_dataset = CNNDataset(VAL_CSV)
test_dataset = CNNDataset(TEST_CSV)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))




train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


features, labels = next(iter(train_loader))

print("\nFirst batch:")
print("Features:", features.shape)
print("Labels:", labels.shape)


model = AudioCNN(
    num_classes=NUM_CLASSES
).to(device)


optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

def train_one_epoch():

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in train_loader:

        features = features.to(device)
        labels = labels.to(device)

        features = features[:, :, :128]

        optimizer.zero_grad()

        output = model(features)

        loss = F.cross_entropy(
            output,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = output.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    average_loss = (
        total_loss / len(train_loader)
    )

    accuracy = correct / total

    return average_loss, accuracy


def evaluate(loader):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(device)
            labels = labels.to(device)

            # Mel features only
            features = features[:, :, :128]

            output = model(features)

            loss = F.cross_entropy(
                output,
                labels
            )

            total_loss += loss.item()

            predictions = output.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = (
        total_loss / len(loader)
    )

    accuracy = correct / total

    return average_loss, accuracy


best_val_accuracy = 0.0

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

for epoch in range(1, EPOCHS + 1):

    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy = evaluate(
        val_loader
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )


    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"  ✓ Best model saved "
            f"(Val Acc: {val_accuracy:.4f})"
        )


print("\nLoading best CNN model...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

test_loss, test_accuracy = evaluate(
    test_loader
)


print("\n==============================")
print("CNN Final Results")
print("==============================")

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

print(
    f"\nBest model saved to: "
    f"{MODEL_PATH}"
)