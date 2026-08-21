import os
import ast
import re
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
    "data/results/zero_shot_tags"
)

RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "zero_shot_tag_results.csv"
)

METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "zero_shot_tag_metrics.csv"
)

EXAMPLES_PATH = os.path.join(
    RESULTS_DIR,
    "zero_shot_tag_examples.csv"
)

MODEL_NAME = (
    "distilbert-base-uncased"
)

MAX_LENGTH = 256

BATCH_SIZE = 16

EMBEDDING_DIM = 256

GRAPH_INPUT_DIM = 140

GRAPH_HIDDEN_DIM = 128

# ------------------------------------------------------------
# Important:
# Use a minimum frequency so that the zero-shot tag vocabulary
# is not dominated by extremely rare MusicCaps phrases.
# ------------------------------------------------------------

MIN_TAG_FREQUENCY = 5

TOP_K_VALUES = [
    1,
    3,
    5
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
# Load dataset
# ============================================================

print(
    "\n============================================================"
)

print(
    "Loading MusicCaps dataset..."
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

print(
    "Columns:",
    list(df.columns)
)


# ============================================================
# Validate columns
# ============================================================

required_columns = [
    "split",
    "caption",
    "tags"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column '{column}' "
            f"not found in dataset."
        )


# ============================================================
# Split dataset
# ============================================================

train_df = (
    df[
        df["split"] == "train"
    ]
    .reset_index(drop=True)
)

test_df = (
    df[
        df["split"] == "test"
    ]
    .reset_index(drop=True)
)

print(
    "\nTraining samples:",
    len(train_df)
)

print(
    "Test samples:",
    len(test_df)
)


# ============================================================
# Parse tags
# ============================================================

def parse_tags(value):

    if isinstance(
        value,
        list
    ):

        return [
            str(tag)
            .strip()
            .lower()

            for tag in value

            if str(tag).strip()
        ]


    if pd.isna(value):

        return []


    value = str(
        value
    ).strip()


    if not value:

        return []


    # --------------------------------------------------------
    # MusicCaps tags are generally stored as Python-list
    # strings.
    # --------------------------------------------------------

    try:

        parsed = ast.literal_eval(
            value
        )

        if isinstance(
            parsed,
            list
        ):

            return [
                str(tag)
                .strip()
                .lower()

                for tag in parsed

                if str(tag).strip()
            ]

    except Exception:

        pass


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    value = value.strip(
        "[]"
    )

    parts = value.split(",")

    return [
        part
        .strip()
        .strip("'")
        .strip('"')
        .lower()

        for part in parts

        if part.strip()
    ]


train_df["parsed_tags"] = (
    train_df["tags"]
    .apply(parse_tags)
)

test_df["parsed_tags"] = (
    test_df["tags"]
    .apply(parse_tags)
)


# ============================================================
# Normalize tag text
# ============================================================

def normalize_tag(
    tag
):

    tag = str(
        tag
    ).lower().strip()

    tag = re.sub(
        r"\s+",
        " ",
        tag
    )

    return tag


# ============================================================
# Build TRAINING-ONLY tag vocabulary
# ============================================================

print(
    "\n============================================================"
)

print(
    "Building training-only tag vocabulary..."
)

print(
    "============================================================"
)


tag_counter = {}


for tags in train_df[
    "parsed_tags"
]:

    for tag in tags:

        tag = normalize_tag(
            tag
        )

        if not tag:

            continue

        if tag not in tag_counter:

            tag_counter[tag] = 0

        tag_counter[tag] += 1


candidate_tags = sorted(
    [
        tag
        for tag, count
        in tag_counter.items()

        if count >= MIN_TAG_FREQUENCY
    ]
)


print(
    "Unique training tags:",
    len(tag_counter)
)

print(
    "Candidate tags after frequency filtering:",
    len(candidate_tags)
)

print(
    "Minimum tag frequency:",
    MIN_TAG_FREQUENCY
)


# ============================================================
# Show most frequent candidate tags
# ============================================================

print(
    "\nMost frequent candidate tags:"
)


most_common_tags = sorted(
    [
        (
            tag,
            count
        )

        for tag, count
        in tag_counter.items()

        if count >= MIN_TAG_FREQUENCY
    ],

    key=lambda x: x[1],

    reverse=True
)


for tag, count in (
    most_common_tags[:30]
):

    print(
        f"  {tag:<45} "
        f"{count}"
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
# Load TRAINED Task 4 dual encoder
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
    "✓ Best dual encoder loaded"
)


# ============================================================
# IMPORTANT:
# We use the TEXT ENCODER FROM TASK 4.
#
# Caption:
# DistilBERT → learned projection → 256-D shared space
#
# Tag:
# DistilBERT → same learned projection → 256-D shared space
#
# This makes the zero-shot tag evaluation consistent with
# the learned Task 4 representation.
# ============================================================

text_encoder = (
    model.text_encoder
)


text_encoder.eval()


# ============================================================
# Encode texts using Task 4 text encoder
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


            # ------------------------------------------------
            # USE THE TRAINED TASK 4 TEXT ENCODER
            # ------------------------------------------------

            embeddings = (
                text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            )


            # TextEncoder already performs L2 normalization,
            # but normalize again defensively.

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


    return torch.cat(
        all_embeddings,
        dim=0
    )


# ============================================================
# Encode candidate tags
# ============================================================

print(
    "\n============================================================"
)

print(
    "Encoding candidate tags in Task 4 shared space..."
)

print(
    "============================================================"
)


tag_embeddings = encode_texts(
    candidate_tags
)


print(
    "\nTag embedding shape:",
    tuple(
        tag_embeddings.shape
    )
)


# ============================================================
# Encode test captions
# ============================================================

print(
    "\n============================================================"
)

print(
    "Encoding test captions in Task 4 shared space..."
)

print(
    "============================================================"
)


test_captions = (
    test_df[
        "caption"
    ]
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
    "Computing caption → tag similarities..."
)

print(
    "============================================================"
)


# Both embeddings are normalized.
#
# Therefore:
#
# cosine similarity = dot product

similarity_matrix = (
    caption_embeddings
    @ tag_embeddings.T
)


print(
    "Similarity matrix:",
    tuple(
        similarity_matrix.shape
    )
)


# ============================================================
# Candidate-tag lookup
# ============================================================

candidate_tag_lookup = {
    normalize_tag(tag)
    for tag in candidate_tags
}


# ============================================================
# Metrics
# ============================================================

def precision_recall_f1_at_k(
    predicted_tags,
    actual_tags,
    k
):

    predicted = set(
        normalize_tag(tag)

        for tag in predicted_tags[:k]
    )


    actual = set(
        normalize_tag(tag)

        for tag in actual_tags

        if normalize_tag(tag)
        in candidate_tag_lookup
    )


    if len(predicted) == 0:

        return (
            0.0,
            0.0,
            0.0
        )


    if len(actual) == 0:

        return (
            0.0,
            0.0,
            0.0
        )


    true_positives = len(
        predicted.intersection(
            actual
        )
    )


    precision = (
        true_positives
        /
        len(predicted)
    )


    recall = (
        true_positives
        /
        len(actual)
    )


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


    return (
        precision,
        recall,
        f1
    )


# ============================================================
# Generate predictions
# ============================================================

print(
    "\n============================================================"
)

print(
    "Generating zero-shot tag predictions..."
)

print(
    "============================================================"
)


all_results = []

qualitative_examples = []


# ------------------------------------------------------------
# Select 10 evenly distributed examples
# ------------------------------------------------------------

example_indices = set(
    np.linspace(
        0,
        len(test_df) - 1,
        10,
        dtype=int
    ).tolist()
)


for index in range(
    len(test_df)
):

    scores = (
        similarity_matrix[index]
    )


    max_k = max(
        TOP_K_VALUES
    )


    top_indices = torch.topk(
        scores,
        k=min(
            max_k,
            len(candidate_tags)
        )
    ).indices.tolist()


    predicted_tags = [

        candidate_tags[i]

        for i in top_indices
    ]


    actual_tags = (
        test_df
        .iloc[index]
        ["parsed_tags"]
    )


    row_result = {

        "index": index,

        "genre": (
            test_df
            .iloc[index]
            ["genre"]
        ),

        "caption": (
            test_df
            .iloc[index]
            ["caption"]
        ),

        "actual_tags": (
            ", ".join(
                actual_tags
            )
        )
    }


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    for k in TOP_K_VALUES:

        precision, recall, f1 = (
            precision_recall_f1_at_k(
                predicted_tags,
                actual_tags,
                k
            )
        )


        row_result[
            f"Precision@{k}"
        ] = precision


        row_result[
            f"Recall@{k}"
        ] = recall


        row_result[
            f"F1@{k}"
        ] = f1


    # --------------------------------------------------------
    # Predicted tags + similarities
    # --------------------------------------------------------

    for rank, tag in enumerate(
        predicted_tags,
        start=1
    ):

        row_result[
            f"Predicted@{rank}"
        ] = tag


        row_result[
            f"Similarity@{rank}"
        ] = float(
            scores[
                top_indices[
                    rank - 1
                ]
            ]
        )


    all_results.append(
        row_result
    )


    # --------------------------------------------------------
    # Qualitative examples
    # --------------------------------------------------------

    if index in example_indices:

        qualitative_examples.append({

            "index": index,

            "genre": (
                test_df
                .iloc[index]
                ["genre"]
            ),

            "caption": (
                test_df
                .iloc[index]
                ["caption"]
            ),

            "actual_tags": (
                ", ".join(
                    actual_tags
                )
            ),

            "predicted_top_5": (
                ", ".join(
                    predicted_tags[:5]
                )
            )
        })


# ============================================================
# Results DataFrame
# ============================================================

results_df = pd.DataFrame(
    all_results
)


# ============================================================
# Overall metrics
# ============================================================

print(
    "\n============================================================"
)

print(
    "ZERO-SHOT TAG PREDICTION RESULTS"
)

print(
    "============================================================"
)


overall_results = []


for k in TOP_K_VALUES:

    precision = (
        results_df[
            f"Precision@{k}"
        ]
        .mean()
    )


    recall = (
        results_df[
            f"Recall@{k}"
        ]
        .mean()
    )


    f1 = (
        results_df[
            f"F1@{k}"
        ]
        .mean()
    )


    overall_results.append({

        "K": k,

        "Precision": precision,

        "Recall": recall,

        "F1": f1
    })


    print(
        f"\nK = {k}"
    )


    print(
        f"Precision@{k}: "
        f"{precision:.4f}"
    )


    print(
        f"Recall@{k}: "
        f"{recall:.4f}"
    )


    print(
        f"F1@{k}: "
        f"{f1:.4f}"
    )


overall_df = pd.DataFrame(
    overall_results
)


# ============================================================
# Qualitative examples
# ============================================================

print(
    "\n============================================================"
)

print(
    "QUALITATIVE ZERO-SHOT TAG EXAMPLES"
)

print(
    "============================================================"
)


for example in (
    qualitative_examples
):

    print(
        "\n------------------------------------------------------------"
    )


    print(
        f"Example index: "
        f"{example['index']}"
    )


    print(
        f"Genre: "
        f"{example['genre']}"
    )


    print(
        "\nCaption:"
    )


    print(
        example["caption"]
    )


    print(
        "\nActual tags:"
    )


    print(
        example["actual_tags"]
    )


    print(
        "\nPredicted top-5 tags:"
    )


    print(
        example["predicted_top_5"]
    )


# ============================================================
# Save results
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


results_df.to_csv(
    RESULTS_PATH,
    index=False
)


overall_df.to_csv(
    METRICS_PATH,
    index=False
)


qualitative_df = pd.DataFrame(
    qualitative_examples
)


qualitative_df.to_csv(
    EXAMPLES_PATH,
    index=False
)


# ============================================================
# Final summary
# ============================================================

print(
    "\n============================================================"
)

print(
    "ZERO-SHOT TAG PREDICTION COMPLETE"
)

print(
    "============================================================"
)


print(
    "\nCandidate tags:",
    len(candidate_tags)
)


print(
    "Minimum tag frequency:",
    MIN_TAG_FREQUENCY
)


print(
    "Test samples:",
    len(test_df)
)


print(
    "\nMetrics saved to:"
)


print(
    METRICS_PATH
)


print(
    "\nDetailed predictions saved to:"
)


print(
    RESULTS_PATH
)


print(
    "\nQualitative examples saved to:"
)


print(
    EXAMPLES_PATH
)


print(
    "\n============================================================"
)