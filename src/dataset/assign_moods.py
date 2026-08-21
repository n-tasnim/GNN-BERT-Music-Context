import ast
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_CSV = "data/processed/splits/fusion_dataset.csv"

OUTPUT_CSV = "data/processed/splits/fusion_dataset_with_mood.csv"


# ============================================================
# Mood definitions
# ============================================================

MOOD_KEYWORDS = {

    "energetic": [
        "energetic",
        "excited",
        "exciting",
        "lively",
        "spirited",
        "powerful",
        "intense",
        "wild energy",
        "hard-hitting",
        "uptempo",
        "upbeat",
        "fast",
        "fast tempo",
        "medium to uptempo",
        "medium fast tempo",
        "danceable",
        "punchy",
        "aggressive",
        "manic",
    ],

    "happy": [
        "happy",
        "cheerful",
        "joyful",
        "positive",
        "uplifting",
        "sunny",
        "peppy",
        "playful",
        "funny",
        "amusing",
        "entertaining",
        "fresh",
        "vibrant",
        "buoyant",
        "chirpy",
        "life is good",
        "enjoy life",
    ],

    "sad": [
        "sad",
        "melancholy",
        "melancholic",
        "heartfelt",
        "sorrow",
        "sorrowful",
        "emotional",
        "sentimental",
        "missing you",
        "love you so badly",
        "dream of you",
    ],

    "calm": [
        "calm",
        "relaxing",
        "relaxed",
        "soothing",
        "tranquil",
        "peaceful",
        "easygoing",
        "mellow",
        "soft",
        "light",
        "gentle",
        "slow",
        "slow tempo",
        "slow to medium tempo",
        "slower to medium",
    ],

    "romantic": [
        "romantic",
        "romance",
        "love song",
        "love performance",
        "love you",
        "missing you",
        "dream of you",
        "special someone",
        "together forever",
        "sensual",
        "passionate",
        "passionate female vocal",
        "heartfelt",
    ],

    "dark": [
        "dark",
        "scary",
        "scary music",
        "spooky",
        "eerie",
        "sinister",
        "haunting",
        "suspense",
        "suspenseful",
        "violent",
        "menacing",
        "ominous",
        "chaotic",
        "chaotic harmony",
    ],

    "neutral": []
}


# ============================================================
# Priority
# ============================================================

# Used when multiple moods receive the same score.
#
# Romantic is checked before sad/calm because words such as
# "love" and "passionate" should generally indicate romance
# rather than simply emotional content.

MOOD_PRIORITY = [
    "romantic",
    "dark",
    "energetic",
    "happy",
    "sad",
    "calm",
    "neutral"
]


# ============================================================
# Parse tags
# ============================================================

def parse_tags(value):

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list):
            return [
                str(tag).lower().strip()
                for tag in parsed
            ]

    except (ValueError, SyntaxError):

        pass

    return []


# ============================================================
# Create searchable text
# ============================================================

def create_search_text(row):

    tags = parse_tags(
        row["tags"]
    )

    caption = str(
        row["caption"]
    ).lower()

    return " ".join(
        tags
    ) + " " + caption


# ============================================================
# Assign mood
# ============================================================

def assign_mood(row):

    tags = parse_tags(
        row["tags"]
    )

    caption = str(
        row["caption"]
    ).lower()

    # Tags are more reliable than free-form caption text,
    # so they receive a higher weight.
    tag_text = " ".join(tags)

    scores = {
        mood: 0
        for mood in MOOD_KEYWORDS
    }

    # --------------------------------------------------------
    # Score tags
    # --------------------------------------------------------

    for mood, keywords in MOOD_KEYWORDS.items():

        for keyword in keywords:

            if keyword in tag_text:
                scores[mood] += 2


    # --------------------------------------------------------
    # Score caption
    # --------------------------------------------------------

    for mood, keywords in MOOD_KEYWORDS.items():

        for keyword in keywords:

            if keyword in caption:
                scores[mood] += 1


    # --------------------------------------------------------
    # Remove neutral from competition
    # --------------------------------------------------------

    non_neutral_scores = {
        mood: score
        for mood, score in scores.items()
        if mood != "neutral"
    }


    # --------------------------------------------------------
    # No mood evidence
    # --------------------------------------------------------

    if max(
        non_neutral_scores.values()
    ) == 0:

        return "neutral"


    # --------------------------------------------------------
    # Find highest score
    # --------------------------------------------------------

    max_score = max(
        non_neutral_scores.values()
    )


    candidates = [
        mood
        for mood, score in non_neutral_scores.items()
        if score == max_score
    ]


    # --------------------------------------------------------
    # Resolve ties using priority
    # --------------------------------------------------------

    for mood in MOOD_PRIORITY:

        if mood in candidates:
            return mood


    return "neutral"


# ============================================================
# Main
# ============================================================

print("Loading fusion dataset...")

df = pd.read_csv(
    INPUT_CSV
)

print(
    "Total samples:",
    len(df)
)


# ============================================================
# Assign moods
# ============================================================

print("\nAssigning mood labels...")

df["mood"] = df.apply(
    assign_mood,
    axis=1
)


# ============================================================
# Save
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# Statistics
# ============================================================

print("\n==============================")
print("MOOD LABELING COMPLETE")
print("==============================")

print(
    "\nMood distribution:"
)

print(
    df["mood"].value_counts()
)


print(
    "\nMood distribution (%):"
)

print(
    (
        df["mood"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_CSV
)