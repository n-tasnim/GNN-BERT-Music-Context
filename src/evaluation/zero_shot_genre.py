import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer

from src.models.dual_encoder import DualEncoder


# ============================================================
# Configuration
# ============================================================

DATASET_CSV = (
    "data/processed/splits/fusion_dataset.csv"
)

MODEL_PATH = (
    "data/models/best_dual_encoder.pt"
)

RESULTS_DIR = (
    "data/results/zero_shot_genre"
)

RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "zero_shot_genre_results.csv"
)

METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "zero_shot_genre_metrics.csv"
)

EXAMPLES_PATH = os.path.join(
    RESULTS_DIR,
    "zero_shot_genre_examples.csv"
)

MODEL_NAME = (
    "distilbert-base-uncased"
)

MAX_LENGTH = 256

BATCH_SIZE = 16

EMBEDDING_DIM = 256

GRAPH_INPUT_DIM = 140

GRAPH_HIDDEN_DIM = 128


# ============================================================
# Genre vocabulary
# ============================================================

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
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# Load dataset
# ============================================================

print(
    "\n============================================================"
)

print(
    "Loading test dataset..."
)

print(
    "============================================================"
)

df = pd.read_csv(
    DATASET_CSV
)

print(
    "Total samples:",
    len(df)
)


# ============================================================
# Validate columns
# ============================================================

required_columns = [
    "split",
    "caption",
    "genre"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column '{column}' "
            f"not found in dataset."
        )


# ============================================================
# Select test split
# ============================================================

test_df = (
    df[
        df["split"] == "test"
    ]
    .reset_index(drop=True)
)

print(
    "\nTest samples:",
    len(test_df)
)


# ============================================================
# Validate genres
# ============================================================

test_df["genre"] = (
    test_df["genre"]
    .astype(str)
    .str.lower()
    .str.strip()
)

unknown_genres = sorted(
    set(test_df["genre"])
    -
    set(GENRES)
)

if unknown_genres:

    raise ValueError(
        "Found genres in test set that are not "
        f"in the zero-shot genre vocabulary: "
        f"{unknown_genres}"
    )


# ============================================================
# Load tokenizer
# ============================================================

print(
    "\n============================================================"
)

print(
    "Loading tokenizer..."
)

print(
    "============================================================"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# Load trained Task 4 dual encoder
# ============================================================

print(
    "\n============================================================"
)

print(
    "Loading trained Task 4 dual encoder..."
)

print(
    "============================================================"
)

model = DualEncoder(
    graph_input_dim=GRAPH_INPUT_DIM,
    graph_hidden_dim=GRAPH_HIDDEN_DIM,
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

print(
    "✓ Best Task 4 dual encoder loaded"
)


# ============================================================
# Use Task 4 text encoder
# ============================================================

text_encoder = (
    model.text_encoder
)

text_encoder.eval()


# ============================================================
# Text encoding function
# ============================================================

def encode_texts(
    texts,
    batch_size=BATCH_SIZE
):

    all_embeddings = []

    with torch.no_grad():

        for start in range(
            0,
            len(texts),
            batch_size
        ):

            end = min(
                start + batch_size,
                len(texts)
            )

            batch_texts = texts[
                start:end
            ]

            encoded = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            input_ids = (
                encoded[
                    "input_ids"
                ]
                .to(device)
            )

            attention_mask = (
                encoded[
                    "attention_mask"
                ]
                .to(device)
            )

            embeddings = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1
            )

            all_embeddings.append(
                embeddings.cpu()
            )

            print(
                f"Encoded "
                f"{end}/{len(texts)}"
            )

    if len(all_embeddings) == 0:

        return torch.empty(
            (0, EMBEDDING_DIM)
        )

    return torch.cat(
        all_embeddings,
        dim=0
    )


# ============================================================
# Encode genre labels
# ============================================================

print(
    "\n============================================================"
)

print(
    "Encoding genre labels..."
)

print(
    "============================================================"
)

genre_embeddings = encode_texts(
    GENRES
)

print(
    "\nGenre embedding shape:",
    tuple(
        genre_embeddings.shape
    )
)


# ============================================================
# Encode test captions
# ============================================================

print(
    "\n============================================================"
)

print(
    "Encoding test captions..."
)

print(
    "============================================================"
)

test_captions = (
    test_df["caption"]
    .fillna("")
    .astype(str)
    .tolist()
)

caption_embeddings = encode_texts(
    test_captions
)

print(
    "\nCaption embedding shape:",
    tuple(
        caption_embeddings.shape
    )
)


# ============================================================
# Similarity matrix
# ============================================================

print(
    "\n============================================================"
)

print(
    "Computing caption → genre similarities..."
)

print(
    "============================================================"
)

similarity_matrix = (
    caption_embeddings
    @ genre_embeddings.T
)

print(
    "Similarity matrix shape:",
    tuple(
        similarity_matrix.shape
    )
)


# ============================================================
# Zero-shot predictions
# ============================================================

print(
    "\n============================================================"
)

print(
    "Generating zero-shot genre predictions..."
)

print(
    "============================================================"
)

predicted_indices = (
    similarity_matrix.argmax(
        dim=1
    )
)

predicted_genres = [
    GENRES[index]
    for index in predicted_indices.tolist()
]


actual_genres = (
    test_df["genre"]
    .tolist()
)


# ============================================================
# Top-k predictions
# ============================================================

top_k = min(
    len(GENRES),
    5
)

top_scores, top_indices = torch.topk(
    similarity_matrix,
    k=top_k,
    dim=1
)


# ============================================================
# Accuracy
# ============================================================

correct = 0

for predicted, actual in zip(
    predicted_genres,
    actual_genres
):

    if predicted == actual:

        correct += 1


accuracy = (
    correct /
    len(actual_genres)
)


# ============================================================
# Per-class metrics
# ============================================================

genre_metrics = []


for genre in GENRES:

    true_positive = 0

    false_positive = 0

    false_negative = 0


    for actual, predicted in zip(
        actual_genres,
        predicted_genres
    ):

        if (
            actual == genre
            and
            predicted == genre
        ):

            true_positive += 1

        elif (
            actual != genre
            and
            predicted == genre
        ):

            false_positive += 1

        elif (
            actual == genre
            and
            predicted != genre
        ):

            false_negative += 1


    if (
        true_positive +
        false_positive
    ) > 0:

        precision = (
            true_positive /
            (
                true_positive +
                false_positive
            )
        )

    else:

        precision = 0.0


    if (
        true_positive +
        false_negative
    ) > 0:

        recall = (
            true_positive /
            (
                true_positive +
                false_negative
            )
        )

    else:

        recall = 0.0


    if (
        precision +
        recall
    ) > 0:

        f1 = (
            2.0
            *
            precision
            *
            recall
            /
            (
                precision +
                recall
            )
        )

    else:

        f1 = 0.0


    genre_metrics.append({

        "Genre": genre,

        "Precision": precision,

        "Recall": recall,

        "F1": f1
    })


# ============================================================
# Macro metrics
# ============================================================

macro_precision = np.mean([
    row["Precision"]
    for row in genre_metrics
])

macro_recall = np.mean([
    row["Recall"]
    for row in genre_metrics
])

macro_f1 = np.mean([
    row["F1"]
    for row in genre_metrics
])


# ============================================================
# Top-k accuracy
# ============================================================

top_k_accuracy = {}


for k in [1, 3, 5]:

    k = min(
        k,
        len(GENRES)
    )

    hits = 0

    for row_index in range(
        len(test_df)
    ):

        actual = actual_genres[
            row_index
        ]

        retrieved_indices = (
            top_indices[
                row_index,
                :k
            ]
            .tolist()
        )

        retrieved_genres = [
            GENRES[index]
            for index in retrieved_indices
        ]

        if actual in retrieved_genres:

            hits += 1


    top_k_accuracy[k] = (
        hits /
        len(test_df)
    )


# ============================================================
# Print final results
# ============================================================

print(
    "\n============================================================"
)

print(
    "ZERO-SHOT GENRE PREDICTION RESULTS"
)

print(
    "============================================================"
)

print(
    f"\nZero-shot Accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Macro Precision: "
    f"{macro_precision:.4f}"
)

print(
    f"Macro Recall: "
    f"{macro_recall:.4f}"
)

print(
    f"Macro F1: "
    f"{macro_f1:.4f}"
)


for k, value in (
    top_k_accuracy.items()
):

    print(
        f"Top-{k} Accuracy: "
        f"{value:.4f}"
    )


# ============================================================
# Create detailed results
# ============================================================

results = []


for index in range(
    len(test_df)
):

    row = test_df.iloc[
        index
    ]

    result = {

        "index": index,

        "genre": row["genre"],

        "caption": row["caption"],

        "predicted_genre": (
            predicted_genres[index]
        ),

        "top_1_similarity": float(
            top_scores[
                index,
                0
            ]
        )
    }


    for rank in range(
        top_k
    ):

        genre_index = (
            top_indices[
                index,
                rank
            ].item()
        )

        result[
            f"top_{rank + 1}_genre"
        ] = GENRES[
            genre_index
        ]

        result[
            f"top_{rank + 1}_similarity"
        ] = float(
            top_scores[
                index,
                rank
            ]
        )


    results.append(
        result
    )


results_df = pd.DataFrame(
    results
)


# ============================================================
# Metrics DataFrame
# ============================================================

overall_metrics = pd.DataFrame([

    {
        "Metric": "Accuracy",
        "Value": accuracy
    },

    {
        "Metric": "Macro Precision",
        "Value": macro_precision
    },

    {
        "Metric": "Macro Recall",
        "Value": macro_recall
    },

    {
        "Metric": "Macro F1",
        "Value": macro_f1
    },

    {
        "Metric": "Top-1 Accuracy",
        "Value": top_k_accuracy[1]
    },

    {
        "Metric": "Top-3 Accuracy",
        "Value": top_k_accuracy[3]
    },

    {
        "Metric": "Top-5 Accuracy",
        "Value": top_k_accuracy[5]
    }

])


# ============================================================
# Qualitative examples
# ============================================================

print(
    "\n============================================================"
)

print(
    "QUALITATIVE ZERO-SHOT GENRE EXAMPLES"
)

print(
    "============================================================"
)


example_count = min(
    10,
    len(test_df)
)


example_indices = np.linspace(
    0,
    len(test_df) - 1,
    num=example_count,
    dtype=int
)


qualitative_examples = []


for example_number, index in enumerate(
    example_indices,
    start=1
):

    row = test_df.iloc[
        index
    ]

    retrieved_genres = []

    retrieved_similarities = []


    for rank in range(
        min(3, top_k)
    ):

        genre_index = (
            top_indices[
                index,
                rank
            ].item()
        )

        retrieved_genres.append(
            GENRES[
                genre_index
            ]
        )

        retrieved_similarities.append(
            float(
                top_scores[
                    index,
                    rank
                ]
            )
        )


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
        "\nActual genre:",
        row["genre"]
    )

    print(
        "\nCaption:"
    )

    print(
        row["caption"]
    )

    print(
        "\nTop-3 zero-shot genres:"
    )

    for rank, (
        genre,
        score
    ) in enumerate(
        zip(
            retrieved_genres,
            retrieved_similarities
        ),
        start=1
    ):

        print(
            f"  {rank}. "
            f"{genre} "
            f"(similarity: "
            f"{score:.4f})"
        )


    qualitative_examples.append({

        "example": example_number,

        "index": index,

        "actual_genre": row["genre"],

        "caption": row["caption"],

        "top_1_genre": (
            retrieved_genres[0]
        ),

        "top_1_similarity": (
            retrieved_similarities[0]
        ),

        "top_2_genre": (
            retrieved_genres[1]
            if len(retrieved_genres) > 1
            else ""
        ),

        "top_2_similarity": (
            retrieved_similarities[1]
            if len(retrieved_similarities) > 1
            else 0.0
        ),

        "top_3_genre": (
            retrieved_genres[2]
            if len(retrieved_genres) > 2
            else ""
        ),

        "top_3_similarity": (
            retrieved_similarities[2]
            if len(retrieved_similarities) > 2
            else 0.0
        )
    })


qualitative_df = pd.DataFrame(
    qualitative_examples
)


# ============================================================
# Per-genre metrics
# ============================================================

per_genre_df = pd.DataFrame(
    genre_metrics
)


# ============================================================
# Save results
# ============================================================

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

overall_metrics.to_csv(
    METRICS_PATH,
    index=False
)

qualitative_df.to_csv(
    EXAMPLES_PATH,
    index=False
)

PER_GENRE_PATH = os.path.join(
    RESULTS_DIR,
    "zero_shot_genre_per_class_metrics.csv"
)

per_genre_df.to_csv(
    PER_GENRE_PATH,
    index=False
)


# ============================================================
# Final summary
# ============================================================

print(
    "\n============================================================"
)

print(
    "ZERO-SHOT GENRE PREDICTION COMPLETE"
)

print(
    "============================================================"
)

print(
    "\nModel:"
)

print(
    "Task 4 trained DualEncoder"
)

print(
    "\nPrediction method:"
)

print(
    "Caption embedding compared against "
    "genre-name embeddings in the Task 4 shared space."
)

print(
    "\nTraining on genre labels:"
)

print(
    "None"
)

print(
    "\nTest samples:",
    len(test_df)
)

print(
    "\nZero-shot Accuracy:",
    f"{accuracy:.4f}"
)

print(
    "Macro Precision:",
    f"{macro_precision:.4f}"
)

print(
    "Macro Recall:",
    f"{macro_recall:.4f}"
)

print(
    "Macro F1:",
    f"{macro_f1:.4f}"
)

print(
    "\nResults saved to:"
)

print(
    RESULTS_PATH
)

print(
    "\nOverall metrics saved to:"
)

print(
    METRICS_PATH
)

print(
    "\nQualitative examples saved to:"
)

print(
    EXAMPLES_PATH
)

print(
    "\nPer-genre metrics saved to:"
)

print(
    PER_GENRE_PATH
)

print(
    "\n============================================================"
)