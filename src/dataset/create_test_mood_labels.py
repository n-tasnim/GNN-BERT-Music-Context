import os
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

MOOD_DATASET = (
    "data/processed/splits/"
    "fusion_dataset_with_mood.csv"
)

EMBEDDINGS_PATH = (
    "data/results/plots/"
    "fusion_tsne_embeddings.npy"
)

OUTPUT_PATH = (
    "data/results/plots/"
    "fusion_tsne_mood_labels.npy"
)


# ============================================================
# Load mood-labelled dataset
# ============================================================

print("Loading mood-labelled fusion dataset...")

df = pd.read_csv(
    MOOD_DATASET
)

print(
    "Total rows:",
    len(df)
)


# ============================================================
# Select fusion TEST split
# ============================================================

test_df = df[
    df["split"] == "test"
].copy()

print(
    "Fusion test samples:",
    len(test_df)
)


# ============================================================
# Load existing embeddings
# ============================================================

print(
    "\nLoading existing fusion embeddings..."
)

embeddings = np.load(
    EMBEDDINGS_PATH
)

print(
    "Embedding shape:",
    embeddings.shape
)


# ============================================================
# Verify sample counts
# ============================================================

if len(test_df) != len(embeddings):

    raise ValueError(
        f"Mismatch!\n"
        f"Test samples: {len(test_df)}\n"
        f"Embeddings: {len(embeddings)}"
    )


# ============================================================
# Create mood labels
# ============================================================

moods = test_df[
    "mood"
].to_numpy()


# ============================================================
# Verify labels
# ============================================================

print(
    "\nMood labels:"
)

print(
    pd.Series(moods).value_counts()
)


print(
    "\nNumber of mood labels:",
    len(moods)
)


# ============================================================
# Save
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

np.save(
    OUTPUT_PATH,
    moods
)


print(
    f"\n✓ Mood labels saved to:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# Final verification
# ============================================================

saved_moods = np.load(
    OUTPUT_PATH,
    allow_pickle=True
)

print(
    "\nSaved label shape:",
    saved_moods.shape
)

print(
    "\n=============================="
)

print(
    "TEST MOOD LABEL ALIGNMENT COMPLETE"
)

print(
    "=============================="
)