import os
import pickle

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import networkx as nx

from transformers import DistilBertTokenizer

from src.models.fusion_model import FusionModel
from src.dataset.fusion_dataloader import load_graph


# ============================================================
# Configuration
# ============================================================

TEST_CSV = "data/processed/splits/fusion_dataset.csv"

MODEL_PATH = "data/models/best_fusion.pt"

OUTPUT_DIR = "data/results/case_studies"

MAX_LENGTH = 256

NUM_CASES = 3

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
# Output directory
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Load dataset
# ============================================================

print("\nLoading fusion dataset...")

df = pd.read_csv(
    TEST_CSV
)

test_df = df[
    df["split"] == "test"
].reset_index(drop=True)

print(
    "Fusion test samples:",
    len(test_df)
)


# ============================================================
# Load tokenizer
# ============================================================

print("\nLoading tokenizer...")

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased"
)


# ============================================================
# Load model
# ============================================================

print("\nLoading trained FusionModel...")

model = FusionModel(
    graph_dim=64,
    text_dim=768,
    attention_dim=128,
    hidden_dim=256,
    num_classes=10,
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
# Select case studies
# ============================================================

# We deliberately choose three different genres
# to demonstrate different audio/text contexts.

selected_indices = [
    0,
    1,
    2
]


# ============================================================
# Helper: load NetworkX graph
# ============================================================

def load_networkx_graph(graph_path):

    with open(
        graph_path,
        "rb"
    ) as f:

        graph = pickle.load(f)

    return graph


# ============================================================
# Helper: select representative graph path
# ============================================================

def select_graph_path(graph):

    """
    Select a representative path through the audio graph.

    This does NOT claim that individual graph nodes correspond
    directly to individual caption words.

    The path is simply a structural visualization of the
    underlying audio graph.
    """

    nodes = list(
        graph.nodes()
    )

    if len(nodes) == 0:

        return []

    if len(nodes) == 1:

        return nodes

    # Find the longest shortest path among graph nodes.
    # This gives a visually meaningful structural path.

    longest_path = []

    for source in nodes:

        for target in nodes:

            if source == target:
                continue

            try:

                path = nx.shortest_path(
                    graph,
                    source=source,
                    target=target
                )

                if len(path) > len(longest_path):

                    longest_path = path

            except nx.NetworkXNoPath:

                continue

    # If no path exists, simply return nodes
    # in graph order.

    if len(longest_path) == 0:

        longest_path = nodes

    return longest_path


# ============================================================
# Helper: normalize attention
# ============================================================

def extract_attention_tokens(
    attention_weights,
    input_ids,
    attention_mask
):

    attention = (
        attention_weights
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    token_ids = (
        input_ids
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    mask = (
        attention_mask
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    tokens = tokenizer.convert_ids_to_tokens(
        token_ids
    )

    valid_tokens = []

    valid_attention = []

    for token, weight, mask_value in zip(
        tokens,
        attention,
        mask
    ):

        if mask_value == 0:
            continue

        if token in [
            "[CLS]",
            "[SEP]",
            "[PAD]"
        ]:
            continue

        valid_tokens.append(
            token
        )

        valid_attention.append(
            float(weight)
        )

    if len(valid_attention) == 0:

        return [], []

    valid_attention = np.asarray(
        valid_attention,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Attention values are sometimes extremely concentrated.
    # Normalize only for visualization.
    # --------------------------------------------------------

    total = valid_attention.sum()

    if total > 0:

        normalized_attention = (
            valid_attention / total
        )

    else:

        normalized_attention = (
            valid_attention
        )

    return (
        valid_tokens,
        normalized_attention
    )


# ============================================================
# Helper: merge WordPiece tokens
# ============================================================

def merge_wordpiece_tokens(
    tokens,
    weights
):

    merged_tokens = []

    merged_weights = []

    current_token = ""
    current_weight = 0.0

    for token, weight in zip(
        tokens,
        weights
    ):

        if token.startswith("##"):

            current_token += token[2:]

            current_weight += weight

        else:

            if current_token != "":

                merged_tokens.append(
                    current_token
                )

                merged_weights.append(
                    current_weight
                )

            current_token = token

            current_weight = weight

    if current_token != "":

        merged_tokens.append(
            current_token
        )

        merged_weights.append(
            current_weight
        )

    return (
        merged_tokens,
        merged_weights
    )


# ============================================================
# Helper: create alignment interpretation
# ============================================================

def create_alignment_summary(
    genre,
    predicted_genre,
    caption,
    top_tokens
):

    token_text = ", ".join(
        top_tokens
    )

    if predicted_genre == genre:

        prediction_text = (
            f"The fusion model correctly predicts the "
            f"{genre} genre."
        )

    else:

        prediction_text = (
            f"The fusion model predicts "
            f"{predicted_genre} instead of the true "
            f"{genre} genre."
        )

    summary = (
        f"{prediction_text} "
        f"The audio graph represents the structural "
        f"characteristics of the selected audio segment, "
        f"while the text branch provides semantic information "
        f"from the caption. "
        f"The strongest attended caption concepts are: "
        f"{token_text}. "
        f"These attended words provide interpretable "
        f"textual context alongside the audio representation."
    )

    return summary


# ============================================================
# Process case studies
# ============================================================

print("\nGenerating case studies...")


for case_number, index in enumerate(
    selected_indices,
    start=1
):

    row = test_df.iloc[index]

    genre = str(
        row["genre"]
    )

    caption = str(
        row["caption"]
    )

    tags = str(
        row.get(
            "tags",
            ""
        )
    )

    mood = str(
        row.get(
            "mood",
            "N/A"
        )
    )

    graph_path = str(
        row["graph_path"]
    )


    print(
        "\n----------------------------------------"
    )

    print(
        f"Case Study {case_number}"
    )

    print(
        "----------------------------------------"
    )

    print(
        "Genre:",
        genre
    )

    print(
        "Mood:",
        mood
    )

    print(
        "Caption:",
        caption
    )

    print(
        "Graph:",
        graph_path
    )


    # ========================================================
    # Load NetworkX graph
    # ========================================================

    networkx_graph = load_networkx_graph(
        graph_path
    )


    # ========================================================
    # Load PyG graph
    # ========================================================

    graph = load_graph(
        graph_path
    )

    # One graph = batch index 0

    graph.batch = torch.zeros(
        graph.x.size(0),
        dtype=torch.long
    )

    graph = graph.to(
        device
    )


    # ========================================================
    # Tokenize caption
    # ========================================================

    encoded = tokenizer(
        caption,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = encoded[
        "input_ids"
    ].to(device)

    attention_mask = encoded[
        "attention_mask"
    ].to(device)


    # ========================================================
    # Model forward pass
    # ========================================================

    with torch.no_grad():

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


    # ========================================================
    # Prediction
    # ========================================================

    prediction = logits.argmax(
        dim=1
    ).item()

    predicted_genre = GENRES[
        prediction
    ]


    print(
        "Predicted genre:",
        predicted_genre
    )


    # ========================================================
    # Attention
    # ========================================================

    (
        valid_tokens,
        valid_attention
    ) = extract_attention_tokens(
        attention_weights,
        input_ids,
        attention_mask
    )


    (
        merged_tokens,
        merged_weights
    ) = merge_wordpiece_tokens(
        valid_tokens,
        valid_attention
    )


    # ========================================================
    # Select top attention tokens
    # ========================================================

    top_k = min(
        8,
        len(merged_tokens)
    )

    if top_k > 0:

        top_indices = np.argsort(
            merged_weights
        )[-top_k:][::-1]

        top_tokens = [
            merged_tokens[i]
            for i in top_indices
        ]

        top_weights = [
            merged_weights[i]
            for i in top_indices
        ]

    else:

        top_tokens = []

        top_weights = []


    print(
        "\nTop attended caption tokens:"
    )

    for token, weight in zip(
        top_tokens,
        top_weights
    ):

        print(
            f"  {token:20s} {weight:.4f}"
        )


    # ========================================================
    # Select representative graph path
    # ========================================================

    graph_path_nodes = select_graph_path(
        networkx_graph
    )


    print(
        "\nRepresentative graph path:"
    )

    print(
        " -> ".join(
            str(node)
            for node in graph_path_nodes
        )
    )


    # ========================================================
    # Create alignment summary
    # ========================================================

    alignment_summary = create_alignment_summary(
        genre=genre,
        predicted_genre=predicted_genre,
        caption=caption,
        top_tokens=top_tokens
    )


    print(
        "\nAlignment interpretation:"
    )

    print(
        alignment_summary
    )


    # ========================================================
    # Create figure
    # ========================================================

    figure = plt.figure(
        figsize=(16, 11)
    )


    # ========================================================
    # Graph visualization
    # ========================================================

    ax_graph = figure.add_axes(
        [0.04, 0.47, 0.43, 0.43]
    )


    # --------------------------------------------------------
    # Graph positions
    # --------------------------------------------------------

    positions = {
        node: (
            i,
            0
        )
        for i, node in enumerate(
            networkx_graph.nodes()
        )
    }


    # --------------------------------------------------------
    # Draw all edges
    # --------------------------------------------------------

    nx.draw_networkx_edges(
        networkx_graph,
        pos=positions,
        ax=ax_graph,
        width=1.5,
        alpha=0.35
    )


    # --------------------------------------------------------
    # Draw all nodes
    # --------------------------------------------------------

    nx.draw_networkx_nodes(
        networkx_graph,
        pos=positions,
        ax=ax_graph,
        node_size=850,
        alpha=0.8
    )


    # --------------------------------------------------------
    # Highlight representative path
    # --------------------------------------------------------

    if len(graph_path_nodes) >= 2:

        path_edges = list(
            zip(
                graph_path_nodes[:-1],
                graph_path_nodes[1:]
            )
        )

        nx.draw_networkx_edges(
            networkx_graph,
            pos=positions,
            edgelist=path_edges,
            ax=ax_graph,
            width=4,
            edge_color="red"
        )


        nx.draw_networkx_nodes(
            networkx_graph,
            pos=positions,
            nodelist=graph_path_nodes,
            ax=ax_graph,
            node_size=1000,
            node_color="orange"
        )


    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    nx.draw_networkx_labels(
        networkx_graph,
        pos=positions,
        ax=ax_graph,
        font_size=9
    )


    ax_graph.set_title(
        "Audio Graph with Representative Path",
        fontsize=14,
        fontweight="bold"
    )


    ax_graph.axis(
        "off"
    )


    # ========================================================
    # Caption / metadata
    # ========================================================

    ax_caption = figure.add_axes(
        [0.52, 0.47, 0.44, 0.43]
    )

    ax_caption.axis(
        "off"
    )


    caption_text = (
        f"True genre: {genre}\n"
        f"Predicted genre: {predicted_genre}\n"
        f"Mood: {mood}\n\n"
        f"Caption:\n"
        f"{caption}\n\n"
        f"Tags:\n"
        f"{tags}\n\n"
        f"Representative graph path:\n"
        f"{' → '.join(str(n) for n in graph_path_nodes)}"
    )


    ax_caption.text(
        0,
        1,
        caption_text,
        va="top",
        fontsize=10,
        wrap=True
    )


    ax_caption.set_title(
        "Caption / Metadata / Prediction",
        fontsize=14,
        loc="left",
        fontweight="bold"
    )


    # ========================================================
    # Attention visualization
    # ========================================================

    ax_attention = figure.add_axes(
        [0.08, 0.23, 0.84, 0.17]
    )


    y_positions = np.arange(
        len(top_tokens)
    )


    ax_attention.barh(
        y_positions,
        top_weights
    )


    ax_attention.set_yticks(
        y_positions
    )

    ax_attention.set_yticklabels(
        top_tokens
    )

    ax_attention.invert_yaxis()


    ax_attention.set_xlabel(
        "Normalized cross-attention"
    )


    ax_attention.set_title(
        "Caption Concepts Most Attended to by the Fusion Model",
        fontsize=13,
        fontweight="bold"
    )


    # ========================================================
    # Alignment interpretation
    # ========================================================

    ax_alignment = figure.add_axes(
        [0.08, 0.04, 0.84, 0.13]
    )

    ax_alignment.axis(
        "off"
    )


    ax_alignment.text(
        0,
        1,
        "Audio–Text Alignment Interpretation:\n"
        + alignment_summary,
        va="top",
        fontsize=10,
        wrap=True
    )


    # ========================================================
    # Main title
    # ========================================================

    figure.suptitle(
        f"Case Study {case_number}: "
        f"{genre.title()}",
        fontsize=19,
        fontweight="bold"
    )


    # ========================================================
    # Save figure
    # ========================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        f"case_study_{case_number}.png"
    )


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"\n✓ Saved: {output_path}"
    )


# ============================================================
# Complete
# ============================================================

print(
    "\n============================================================"
)

print(
    "CASE STUDIES COMPLETE"
)

print(
    "============================================================"
)