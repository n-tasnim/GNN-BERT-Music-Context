import os

import numpy as np
import pandas as pd
import torch

from src.models.dual_encoder import DualEncoder
from src.dataset.dual_encoder_dataloader import (
    create_dual_encoder_dataloaders
)


# ============================================================
# Configuration
# ============================================================

TEST_CSV = (
    "data/processed/splits/fusion_dataset.csv"
)

MODEL_PATH = (
    "data/models/best_dual_encoder.pt"
)

RESULTS_DIR = (
    "data/results/retrieval"
)

RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "retrieval_results.csv"
)

BATCH_SIZE = 8

EMBEDDING_DIM = 256


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# Create test DataLoader
# ============================================================

print("\nLoading test dataset...")

_, _, test_loader = (
    create_dual_encoder_dataloaders(
        TEST_CSV,
        batch_size=BATCH_SIZE
    )
)

print(
    "Test samples:",
    len(test_loader.dataset)
)


# ============================================================
# Load model
# ============================================================

print(
    "\nLoading best dual encoder..."
)

model = DualEncoder(
    graph_input_dim=140,
    graph_hidden_dim=128,
    embedding_dim=EMBEDDING_DIM,
    freeze_bert=True
).to(device)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("✓ Dual encoder loaded")


# ============================================================
# Generate test embeddings
# ============================================================

print(
    "\nGenerating test embeddings..."
)

all_graph_embeddings = []
all_text_embeddings = []


with torch.no_grad():

    for batch_idx, batch in enumerate(
        test_loader,
        start=1
    ):

        graph = batch[
            "graph"
        ].to(device)

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)


        # ----------------------------------------------------
        # Dual encoder
        # ----------------------------------------------------

        graph_embedding, text_embedding = (
            model(
                graph=graph,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        )


        # ----------------------------------------------------
        # Store embeddings
        # ----------------------------------------------------

        all_graph_embeddings.append(
            graph_embedding.cpu()
        )

        all_text_embeddings.append(
            text_embedding.cpu()
        )


        print(
            f"Processed batch "
            f"{batch_idx}/{len(test_loader)}"
        )


# ============================================================
# Stack embeddings
# ============================================================

graph_embeddings = torch.cat(
    all_graph_embeddings,
    dim=0
)

text_embeddings = torch.cat(
    all_text_embeddings,
    dim=0
)


print(
    "\nGraph embedding shape:",
    tuple(graph_embeddings.shape)
)

print(
    "Text embedding shape:",
    tuple(text_embeddings.shape)
)


# ============================================================
# Similarity matrix
# ============================================================

print(
    "\nComputing cross-modal similarities..."
)

# The DualEncoder already L2-normalizes both
# graph and text embeddings.
#
# Therefore:
#
# cosine similarity = graph_embedding · text_embedding

similarity_matrix = (
    text_embeddings @ graph_embeddings.T
)


print(
    "Similarity matrix shape:",
    tuple(
        similarity_matrix.shape
    )
)


# ============================================================
# Recall@K
# ============================================================

def recall_at_k(
    similarity,
    k
):

    num_samples = similarity.size(0)

    top_k_indices = torch.topk(
        similarity,
        k=k,
        dim=1
    ).indices

    correct = torch.arange(
        num_samples
    ).unsqueeze(1)

    hits = (
        top_k_indices == correct
    ).any(dim=1)

    return hits.float().mean().item()


# ============================================================
# Caption → Audio
# ============================================================

print(
    "\n============================================================"
)

print(
    "CAPTION → AUDIO RETRIEVAL"
)

print(
    "============================================================"
)


caption_to_audio_r1 = recall_at_k(
    similarity_matrix,
    1
)

caption_to_audio_r5 = recall_at_k(
    similarity_matrix,
    5
)

caption_to_audio_r10 = recall_at_k(
    similarity_matrix,
    10
)


print(
    f"R@1 : {caption_to_audio_r1:.4f}"
)

print(
    f"R@5 : {caption_to_audio_r5:.4f}"
)

print(
    f"R@10: {caption_to_audio_r10:.4f}"
)


# ============================================================
# Audio → Caption
# ============================================================

print(
    "\n============================================================"
)

print(
    "AUDIO → CAPTION RETRIEVAL"
)

print(
    "============================================================"
)


audio_to_caption_r1 = recall_at_k(
    similarity_matrix.T,
    1
)

audio_to_caption_r5 = recall_at_k(
    similarity_matrix.T,
    5
)

audio_to_caption_r10 = recall_at_k(
    similarity_matrix.T,
    10
)


print(
    f"R@1 : {audio_to_caption_r1:.4f}"
)

print(
    f"R@5 : {audio_to_caption_r5:.4f}"
)

print(
    f"R@10: {audio_to_caption_r10:.4f}"
)


# ============================================================
# Final retrieval table
# ============================================================

results = pd.DataFrame(
    {
        "Direction": [
            "Caption → Audio",
            "Audio → Caption"
        ],
        "R@1": [
            caption_to_audio_r1,
            audio_to_caption_r1
        ],
        "R@5": [
            caption_to_audio_r5,
            audio_to_caption_r5
        ],
        "R@10": [
            caption_to_audio_r10,
            audio_to_caption_r10
        ]
    }
)


print(
    "\n============================================================"
)

print(
    "TASK 4 RETRIEVAL RESULTS"
)

print(
    "============================================================"
)

print(
    results.to_string(
        index=False,
        formatters={
            "R@1": "{:.4f}".format,
            "R@5": "{:.4f}".format,
            "R@10": "{:.4f}".format
        }
    )
)


# ============================================================
# Save results
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


results.to_csv(
    RESULTS_PATH,
    index=False
)


# ============================================================
# Save embeddings
# ============================================================

np.save(
    os.path.join(
        RESULTS_DIR,
        "test_graph_embeddings.npy"
    ),
    graph_embeddings.numpy()
)

np.save(
    os.path.join(
        RESULTS_DIR,
        "test_text_embeddings.npy"
    ),
    text_embeddings.numpy()
)


print(
    f"\n✓ Retrieval results saved to:"
)

print(
    RESULTS_PATH
)

print(
    "\n✓ Test embeddings saved."
)


print(
    "\n============================================================"
)

print(
    "RETRIEVAL EVALUATION COMPLETE"
)

print(
    "============================================================"
)