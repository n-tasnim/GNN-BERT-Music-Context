import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ============================================================
# Configuration
# ============================================================

EMBEDDING_PATH = (
    "data/results/plots/"
    "fusion_tsne_embeddings.npy"
)

MOOD_LABEL_PATH = (
    "data/results/plots/"
    "fusion_tsne_mood_labels.npy"
)

OUTPUT_PATH = (
    "data/results/plots/"
    "fusion_tsne_mood.png"
)


MOODS = [
    "energetic",
    "romantic",
    "calm",
    "neutral",
    "happy",
    "dark",
    "sad"
]


# ============================================================
# Create output directory
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)


# ============================================================
# Load embeddings
# ============================================================

print("Loading fused representations...")

embeddings = np.load(
    EMBEDDING_PATH
)

print(
    "Embedding shape:",
    embeddings.shape
)


# ============================================================
# Load mood labels
# ============================================================

print("\nLoading mood labels...")

moods = np.load(
    MOOD_LABEL_PATH,
    allow_pickle=True
)

print(
    "Mood label shape:",
    moods.shape
)


# ============================================================
# Verify alignment
# ============================================================

if len(embeddings) != len(moods):

    raise ValueError(
        "Number of embeddings and mood labels "
        "does not match."
    )


print(
    "\nSamples:",
    len(embeddings)
)


# ============================================================
# Run t-SNE
# ============================================================

print("\nRunning t-SNE...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
)

embeddings_2d = tsne.fit_transform(
    embeddings
)

print(
    "t-SNE output shape:",
    embeddings_2d.shape
)


# ============================================================
# Plot
# ============================================================

print("\nCreating mood t-SNE plot...")

plt.figure(
    figsize=(12, 9)
)


unique_moods = [
    mood
    for mood in MOODS
    if mood in moods
]


for mood in unique_moods:

    indices = (
        moods == mood
    )

    plt.scatter(
        embeddings_2d[indices, 0],
        embeddings_2d[indices, 1],
        label=mood,
        alpha=0.75,
        s=60
    )


# ============================================================
# Labels
# ============================================================

plt.title(
    "t-SNE of GNN–BERT Fused Representations by Mood",
    fontsize=16
)

plt.xlabel(
    "t-SNE Dimension 1"
)

plt.ylabel(
    "t-SNE Dimension 2"
)

plt.legend(
    title="Mood",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.grid(
    alpha=0.2
)

plt.tight_layout()


# ============================================================
# Save
# ============================================================

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"\n✓ Mood t-SNE saved to:"
)

print(
    OUTPUT_PATH
)

print(
    "\n=============================="
)

print(
    "MOOD t-SNE COMPLETE"
)

print(
    "=============================="
)