import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from src.dataset.fusion_dataloader import create_fusion_dataloader
from src.models.fusion_model import FusionModel


# ============================================================
# Configuration
# ============================================================

TEST_CSV = "data/processed/splits/fusion_dataset.csv"
MODEL_PATH = "data/models/best_fusion.pt"

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

RESULTS_DIR = "data/results/plots"

TSNE_PATH = os.path.join(
    RESULTS_DIR,
    "fusion_tsne_genre.png"
)

EMBEDDING_PATH = os.path.join(
    RESULTS_DIR,
    "fusion_tsne_embeddings.npy"
)

LABEL_PATH = os.path.join(
    RESULTS_DIR,
    "fusion_tsne_labels.npy"
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Create results directory
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


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
# Load trained fusion model
# ============================================================

print("\nLoading trained fusion model...")

model = FusionModel(
    graph_dim=64,
    text_dim=768,
    attention_dim=128,
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
# Extract fused representations z
# ============================================================

all_embeddings = []
all_labels = []


print("\nExtracting fused representations...")

with torch.no_grad():

    for batch in test_loader:

        graph = batch["graph"].to(device)

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["labels"].to(device)

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        (
            logits,
            graph_embedding,
            text_hidden_states,
            attention_weights,
            z
        ) = model(
            graph,
            input_ids,
            attention_mask
        )

        # ----------------------------------------------------
        # Save z
        # ----------------------------------------------------

        all_embeddings.append(
            z.cpu().numpy()
        )

        all_labels.append(
            labels.cpu().numpy()
        )


# ============================================================
# Convert to NumPy
# ============================================================

all_embeddings = np.concatenate(
    all_embeddings,
    axis=0
)

all_labels = np.concatenate(
    all_labels,
    axis=0
)


print(
    "\nFused representation shape:",
    all_embeddings.shape
)

print(
    "Labels shape:",
    all_labels.shape
)


# ============================================================
# Save raw embeddings
# ============================================================

np.save(
    EMBEDDING_PATH,
    all_embeddings
)

np.save(
    LABEL_PATH,
    all_labels
)

print(
    f"✓ Embeddings saved to: {EMBEDDING_PATH}"
)

print(
    f"✓ Labels saved to: {LABEL_PATH}"
)


# ============================================================
# t-SNE
# ============================================================

print("\nRunning t-SNE...")

# Perplexity must be smaller than the number
# of test samples.
#
# We have 77 test samples, so 30 is safe.

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
)

embeddings_2d = tsne.fit_transform(
    all_embeddings
)


print(
    "t-SNE output shape:",
    embeddings_2d.shape
)


# ============================================================
# Plot
# ============================================================

print("\nCreating t-SNE plot...")

plt.figure(
    figsize=(12, 9)
)

unique_labels = sorted(
    np.unique(all_labels)
)

for label in unique_labels:

    indices = (
        all_labels == label
    )

    plt.scatter(
        embeddings_2d[indices, 0],
        embeddings_2d[indices, 1],
        label=GENRES[label],
        alpha=0.75,
        s=60
    )


plt.title(
    "t-SNE of GNN–BERT Fused Representations",
    fontsize=16
)

plt.xlabel(
    "t-SNE Dimension 1"
)

plt.ylabel(
    "t-SNE Dimension 2"
)

plt.legend(
    title="Genre",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.grid(
    alpha=0.2
)

plt.tight_layout()


# ============================================================
# Save plot
# ============================================================

plt.savefig(
    TSNE_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"\n✓ t-SNE plot saved to: {TSNE_PATH}"
)

print("\n==============================")
print("t-SNE COMPLETE")
print("==============================")