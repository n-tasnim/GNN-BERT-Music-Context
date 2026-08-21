import ast
import random

import pandas as pd


# ============================================================
# Paths
# ============================================================

GRAPH_TRAIN_CSV = "data/processed/splits/graph_train.csv"
GRAPH_VAL_CSV = "data/processed/splits/graph_val.csv"
GRAPH_TEST_CSV = "data/processed/splits/graph_test.csv"

MUSICCPS_CSV = "data/raw/text/musiccaps-public.csv"

OUTPUT_CSV = "data/processed/splits/fusion_dataset.csv"


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# Genre keywords
# ============================================================

GENRE_KEYWORDS = {
    "blues": [
        "blues",
    ],

    "classical": [
        "classical",
    ],

    "country": [
        "country",
        "country music",
        "country pop",
    ],

    "disco": [
        "disco",
    ],

    "hiphop": [
        "hip hop",
        "hip-hop",
        "hiphop",
        "rap",
    ],

    "jazz": [
        "jazz",
        "jazzy",
    ],

    "metal": [
        "metal",
        "heavy metal",
    ],

    "pop": [
        "pop",
        "pop music",
    ],

    "reggae": [
        "reggae",
        "reggae music",
    ],

    "rock": [
        "rock",
        "rock music",
    ],
}


# ============================================================
# Load GTZAN graph splits
# ============================================================

print("Loading GTZAN graph splits...")

train_df = pd.read_csv(GRAPH_TRAIN_CSV)
val_df = pd.read_csv(GRAPH_VAL_CSV)
test_df = pd.read_csv(GRAPH_TEST_CSV)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

graph_df = pd.concat(
    [
        train_df,
        val_df,
        test_df,
    ],
    ignore_index=True
)

print(
    "Total graph samples:",
    len(graph_df)
)


# ============================================================
# Load MusicCaps
# ============================================================

print("\nLoading MusicCaps...")

musiccaps_df = pd.read_csv(
    MUSICCPS_CSV
)

print(
    "Total MusicCaps samples:",
    len(musiccaps_df)
)


# ============================================================
# Parse aspect_list
# ============================================================

def parse_tags(value):

    if pd.isna(value):
        return []

    try:
        return ast.literal_eval(value)

    except (ValueError, SyntaxError):

        return []


musiccaps_df["tags"] = (
    musiccaps_df["aspect_list"]
    .apply(parse_tags)
)


# ============================================================
# Assign genre to MusicCaps
# ============================================================

def detect_genre(row):

    tags = [
        str(tag).lower()
        for tag in row["tags"]
    ]

    caption = str(
        row["caption"]
    ).lower()


    combined_text = (
        " ".join(tags)
        + " "
        + caption
    )

   
    strong_keywords = {
        "hiphop": [
            "hip hop",
            "hip-hop",
            "hiphop",
            "rap music",
            "rap song",
        ],

        "country": [
            "country music",
            "country song",
            "country pop",
        ],

        "classical": [
            "classical music",
            "classical piece",
            "classical song",
        ],

        "reggae": [
            "reggae music",
            "reggae song",
            "reggae piece",
        ],

        "disco": [
            "disco music",
            "disco song",
            "disco piece",
        ],

        "blues": [
            "blues music",
            "blues song",
            "blues piece",
        ],

        "jazz": [
            "jazz music",
            "jazz song",
            "jazz piece",
            "jazz performance",
        ],

        "metal": [
            "heavy metal",
            "metal music",
            "metal song",
            "metal piece",
        ],

        "rock": [
            "rock music",
            "rock song",
            "rock piece",
        ],

        "pop": [
            "pop music",
            "pop song",
            "pop piece",
        ],
    }

    for genre, keywords in strong_keywords.items():

        for keyword in keywords:

            if keyword in combined_text:

                return genre

    fallback_keywords = {
        "hiphop": ["hip hop", "hip-hop", "hiphop"],
        "country": ["country"],
        "classical": ["classical"],
        "reggae": ["reggae"],
        "disco": ["disco"],
        "blues": ["blues"],
        "jazz": ["jazz", "jazzy"],
        "metal": ["metal"],
        "rock": ["rock"],
        "pop": ["pop"],
    }

    matches = []

    for genre, keywords in fallback_keywords.items():

        for keyword in keywords:

            if keyword in combined_text:

                matches.append(genre)

                break

    if matches:

        return matches[0]

    return None


musiccaps_df["genre"] = (
    musiccaps_df.apply(
        detect_genre,
        axis=1
    )
)


musiccaps_genre_df = musiccaps_df[
    musiccaps_df["genre"].notna()
].copy()


print(
    "MusicCaps samples with genre:",
    len(musiccaps_genre_df)
)


# ============================================================
# Remove duplicate YouTube IDs
# ============================================================

musiccaps_genre_df = (
    musiccaps_genre_df
    .drop_duplicates(
        subset=["ytid"]
    )
)


# ============================================================
# Create genre-specific pools
# ============================================================

genre_pools = {}

for genre in GENRE_KEYWORDS:

    candidates = (
        musiccaps_genre_df[
            musiccaps_genre_df["genre"] == genre
        ]
        .copy()
    )

    # Shuffle deterministically.

    candidates = candidates.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(
        drop=True
    )

    genre_pools[genre] = candidates


# ============================================================
# Show available candidates
# ============================================================

print("\nAvailable MusicCaps samples by genre:")

for genre, candidates in genre_pools.items():

    print(
        f"{genre:10s}: {len(candidates)}"
    )


# ============================================================
# Create unique graph-caption pairs
# ============================================================

used_ytids = set()

fusion_rows = []

failed_matches = []


for _, graph_row in graph_df.iterrows():

    genre = graph_row["genre"]

    candidates = genre_pools.get(
        genre,
        pd.DataFrame()
    )

    selected_row = None

    # Find the first unused caption.

    for _, candidate in candidates.iterrows():

        ytid = candidate["ytid"]

        if ytid not in used_ytids:

            selected_row = candidate

            break

    # If no unique caption exists,
    # record the failure.

    if selected_row is None:

        failed_matches.append(
            genre
        )

        continue

    used_ytids.add(
        selected_row["ytid"]
    )

    fusion_rows.append(
        {
            "split": graph_row["split"],

            "genre": genre,

            "graph_path": graph_row[
                "graph_path"
            ],

            "feature_path": graph_row[
                "feature_path"
            ],

            "ytid": selected_row[
                "ytid"
            ],

            "start_s": selected_row[
                "start_s"
            ],

            "end_s": selected_row[
                "end_s"
            ],

            "caption": selected_row[
                "caption"
            ],

            "tags": selected_row[
                "tags"
            ],
        }
    )


# ============================================================
# Create DataFrame
# ============================================================

fusion_df = pd.DataFrame(
    fusion_rows
)


# ============================================================
# Report matching problems
# ============================================================

if failed_matches:

    print(
        "\nWARNING:"
    )

    print(
        "Could not find unique MusicCaps "
        "captions for:"
    )

    for genre in failed_matches:

        print(
            " ",
            genre
        )


# ============================================================
# Save
# ============================================================

fusion_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# Final statistics
# ============================================================

print(
    "\n=============================="
)

print(
    "Fusion Dataset Created"
)

print(
    "=============================="
)

print(
    "Total samples:",
    len(fusion_df)
)

print(
    "\nSamples by split:"
)

print(
    fusion_df["split"].value_counts()
)


print(
    "\nSamples by genre:"
)

print(
    fusion_df["genre"].value_counts()
)


print(
    "\nUnique MusicCaps captions:"
)

print(
    fusion_df["ytid"].nunique()
)


print(
    "\nDuplicate MusicCaps captions:"
)

print(
    len(fusion_df)
    - fusion_df["ytid"].nunique()
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_CSV
)