import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

from src.models.dual_encoder import DualEncoder


# ============================================================
# Configuration
# ============================================================

TEST_CSV = (
    "data/processed/splits/"
    "fusion_dataset.csv"
)

MODEL_PATH = (
    "data/models/"
    "best_dual_encoder.pt"
)

EMBEDDINGS_DIR = (
    "data/results/"
    "retrieval"
)

OUTPUT_DIR = (
    "data/results/"
    "qualitative_retrieval"
)

GRAPH_EMBEDDINGS_PATH = os.path.join(
    EMBEDDINGS_DIR,
    "test_graph_embeddings.npy"
)

TEXT_EMBEDDINGS_PATH = os.path.join(
    EMBEDDINGS_DIR,
    "test_text_embeddings.npy"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "qualitative_retrieval_examples.csv"
)

BATCH_SIZE = 8

MAX_LENGTH = 256

EMBEDDING_DIM = 256

NUM_EXAMPLES = 10

TOP_K = 3


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
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


# ============================================================
# Output directory
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Load test dataset
# ============================================================

print(
    "\nLoading test dataset..."
)

df = pd.read_csv(
    TEST_CSV
)

test_df = df[
    df["split"] == "test"
].reset_index(drop=True)

print(
    "Test samples:",
    len(test_df)
)


# ============================================================
# Check number of examples
# ============================================================

num_examples = min(
    NUM_EXAMPLES,
    len(test_df)
)


# ============================================================
# Load saved embeddings
# ============================================================

print(
    "\nLoading saved test embeddings..."
)


graph_embeddings = torch.tensor(
    np.load(
        GRAPH_EMBEDDINGS_PATH
    ),
    dtype=torch.float32
)


text_embeddings = torch.tensor(
    np.load(
        TEXT_EMBEDDINGS_PATH
    ),
    dtype=torch.float32
)


print(
    "Graph embeddings:",
    tuple(graph_embeddings.shape)
)

print(
    "Text embeddings:",
    tuple(text_embeddings.shape)
)


# ============================================================
# Verify embedding count
# ============================================================

if len(graph_embeddings) != len(test_df):

    raise ValueError(
        "Number of graph embeddings does not match "
        "number of test samples."
    )


if len(text_embeddings) != len(test_df):

    raise ValueError(
        "Number of text embeddings does not match "
        "number of test samples."
    )


# ============================================================
# Normalize embeddings
# ============================================================

# The embeddings were already normalized by DualEncoder.
# We normalize again here only as a safety measure before
# calculating cosine similarity.

graph_embeddings = F.normalize(
    graph_embeddings,
    p=2,
    dim=1
)

text_embeddings = F.normalize(
    text_embeddings,
    p=2,
    dim=1
)


# ============================================================
# Compute similarity matrix
# ============================================================

print(
    "\nComputing cross-modal similarities..."
)


similarity_matrix = (
    text_embeddings
    @ graph_embeddings.T
)


print(
    "Similarity matrix:",
    tuple(
        similarity_matrix.shape
    )
)


# ============================================================
# Select query examples
# ============================================================

# We choose 10 different test examples spread across
# the test set rather than simply taking the first 10.

query_indices = np.linspace(
    0,
    len(test_df) - 1,
    num=num_examples,
    dtype=int
)


# Prevent accidental duplicates if the test set is very small.

query_indices = list(
    dict.fromkeys(
        query_indices.tolist()
    )
)


print(
    "\nSelected query indices:",
    query_indices
)


# ============================================================
# Helper function
# ============================================================

def clean_filename(path):

    if pd.isna(path):

        return "N/A"

    return os.path.basename(
        str(path)
    )


# ============================================================
# Generate qualitative retrieval examples
# ============================================================

print(
    "\n============================================================"
)

print(
    "GENERATING QUALITATIVE RETRIEVAL EXAMPLES"
)

print(
    "============================================================"
)


results = []


for example_number, query_index in enumerate(
    query_indices,
    start=1
):

    query_row = test_df.iloc[
        query_index
    ]


    query_caption = str(
        query_row["caption"]
    )

    query_genre = str(
        query_row["genre"]
    )

    query_graph_path = str(
        query_row["graph_path"]
    )


    # --------------------------------------------------------
    # Similarities for this caption
    # --------------------------------------------------------

    similarities = (
        similarity_matrix[
            query_index
        ]
    )


    # --------------------------------------------------------
    # Retrieve top-K audio clips
    # --------------------------------------------------------

    top_scores, top_indices = torch.topk(
        similarities,
        k=min(
            TOP_K,
            len(test_df)
        )
    )


    top_indices = (
        top_indices
        .cpu()
        .numpy()
    )

    top_scores = (
        top_scores
        .cpu()
        .numpy()
    )


    # --------------------------------------------------------
    # Print query
    # --------------------------------------------------------

    print(
        "\n------------------------------------------------------------"
    )

    print(
        f"Example {example_number}"
    )

    print(
        "------------------------------------------------------------"
    )

    print(
        "\nQuery genre:",
        query_genre
    )

    print(
        "\nQuery caption:"
    )

    print(
        query_caption
    )

    print(
        "\nQuery audio:"
    )

    print(
        clean_filename(
            query_graph_path
        )
    )

    print(
        "\nTop-3 retrieved audio:"
    )


    # --------------------------------------------------------
    # Store retrieval results
    # --------------------------------------------------------

    example_result = {
        "example": example_number,
        "query_index": query_index,
        "query_genre": query_genre,
        "query_caption": query_caption,
        "query_audio": clean_filename(
            query_graph_path
        )
    }


    for rank, (
        retrieved_index,
        score
    ) in enumerate(
        zip(
            top_indices,
            top_scores
        ),
        start=1
    ):

        retrieved_row = test_df.iloc[
            retrieved_index
        ]


        retrieved_genre = str(
            retrieved_row["genre"]
        )

        retrieved_caption = str(
            retrieved_row["caption"]
        )

        retrieved_graph_path = str(
            retrieved_row["graph_path"]
        )


        retrieved_audio = clean_filename(
            retrieved_graph_path
        )


        print(
            f"\n  Rank {rank}"
        )

        print(
            f"    Audio: {retrieved_audio}"
        )

        print(
            f"    Genre: {retrieved_genre}"
        )

        print(
            f"    Similarity: {score:.4f}"
        )

        print(
            f"    Caption: {retrieved_caption}"
        )


        # ----------------------------------------------------
        # Save rank-specific information
        # ----------------------------------------------------

        example_result[
            f"rank_{rank}_audio"
        ] = retrieved_audio

        example_result[
            f"rank_{rank}_genre"
        ] = retrieved_genre

        example_result[
            f"rank_{rank}_similarity"
        ] = float(score)

        example_result[
            f"rank_{rank}_caption"
        ] = retrieved_caption


    results.append(
        example_result
    )


# ============================================================
# Create results dataframe
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# Save CSV
# ============================================================

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# Summary statistics
# ============================================================

print(
    "\n============================================================"
)

print(
    "QUALITATIVE RETRIEVAL SUMMARY"
)

print(
    "============================================================"
)


print(
    f"Examples generated: "
    f"{len(results_df)}"
)


print(
    f"Top-K retrieved per query: "
    f"{TOP_K}"
)


print(
    f"Results saved to:"
)

print(
    OUTPUT_CSV
)


# ============================================================
# Genre agreement analysis
# ============================================================

print(
    "\n============================================================"
)

print(
    "TOP-1 GENRE AGREEMENT"
)

print(
    "============================================================"
)


genre_matches = 0


for result in results:

    query_genre = result[
        "query_genre"
    ]

    retrieved_genre = result[
        "rank_1_genre"
    ]


    if query_genre == retrieved_genre:

        genre_matches += 1


genre_agreement = (
    genre_matches /
    len(results)
)


print(
    f"Top-1 retrieved genre matches "
    f"query genre in "
    f"{genre_matches}/{len(results)} "
    f"cases "
    f"({genre_agreement:.2%})"
)


# ============================================================
# Display compact report
# ============================================================

print(
    "\n============================================================"
)

print(
    "COMPACT RETRIEVAL TABLE"
)

print(
    "============================================================"
)


compact_rows = []


for result in results:

    compact_rows.append(
        {
            "Example": result[
                "example"
            ],

            "Query Genre": result[
                "query_genre"
            ],

            "Top-1 Genre": result[
                "rank_1_genre"
            ],

            "Top-1 Similarity": result[
                "rank_1_similarity"
            ],

            "Top-2 Genre": result[
                "rank_2_genre"
            ],

            "Top-2 Similarity": result[
                "rank_2_similarity"
            ],

            "Top-3 Genre": result[
                "rank_3_genre"
            ],

            "Top-3 Similarity": result[
                "rank_3_similarity"
            ]
        }
    )


compact_df = pd.DataFrame(
    compact_rows
)


print(
    compact_df.to_string(
        index=False,
        formatters={
            "Top-1 Similarity":
                "{:.4f}".format,

            "Top-2 Similarity":
                "{:.4f}".format,

            "Top-3 Similarity":
                "{:.4f}".format
        }
    )
)


# ============================================================
# Complete
# ============================================================

print(
    "\n============================================================"
)

print(
    "QUALITATIVE RETRIEVAL COMPLETE"
)

print(
    "============================================================"
)