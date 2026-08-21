import os

import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.dataset.dataloader import create_graph_dataloaders
from src.models.graphsage import GraphSAGE


# ============================================================
# Configuration
# ============================================================

TRAIN_CSV = "data/processed/splits/graph_train.csv"
VAL_CSV = "data/processed/splits/graph_val.csv"
TEST_CSV = "data/processed/splits/graph_test.csv"

MODEL_DIR = "data/models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_graphsage.pt"
)

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30

INPUT_DIM = 140
HIDDEN_DIM = 64
NUM_CLASSES = 10


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


train_loader, val_loader, test_loader = create_graph_dataloaders(
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    batch_size=BATCH_SIZE
)

print("Training graphs:", len(train_loader.dataset))
print("Validation graphs:", len(val_loader.dataset))
print("Test graphs:", len(test_loader.dataset))



model = GraphSAGE(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
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

    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        
        output = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        
        loss = F.cross_entropy(
            output,
            batch.y
        )


        loss.backward()


        optimizer.step()

        total_loss += loss.item()

        predictions = output.argmax(dim=1)

        correct += (
            predictions == batch.y
        ).sum().item()

        total += batch.y.size(0)

    average_loss = total_loss / len(train_loader)

    accuracy = correct / total

    return average_loss, accuracy


def evaluate(loader):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            output = model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            loss = F.cross_entropy(
                output,
                batch.y
            )

            total_loss += loss.item()

            predictions = output.argmax(dim=1)

            correct += (
                predictions == batch.y
            ).sum().item()

            total += batch.y.size(0)

    average_loss = total_loss / len(loader)

    accuracy = correct / total

    return average_loss, accuracy


best_val_accuracy = 0.0

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


for epoch in range(1, EPOCHS + 1):

    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy = evaluate(
        val_loader
    )

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

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


print("\nLoading best model...")

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
print("Final Results")
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