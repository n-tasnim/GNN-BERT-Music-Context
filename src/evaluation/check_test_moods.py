import pandas as pd

INPUT_CSV = "data/processed/splits/fusion_dataset_with_mood.csv"

df = pd.read_csv(INPUT_CSV)

# Only the exact fusion test set
test_df = df[df["split"] == "test"].copy()

print("\n" + "=" * 60)
print("FUSION TEST-SET MOOD DISTRIBUTION")
print("=" * 60)

print(
    test_df["mood"].value_counts()
)

print("\nPercentages:")
print(
    (
        test_df["mood"]
        .value_counts(normalize=True) * 100
    ).round(2)
)

print("\nTotal test samples:", len(test_df))


# ============================================================
# Show every test sample
# ============================================================

print("\n" + "=" * 60)
print("TEST SAMPLES AND ASSIGNED MOODS")
print("=" * 60)

for _, row in test_df.iterrows():

    print("\n----------------------------------------")

    print(
        "Genre:",
        row["genre"]
    )

    print(
        "Mood:",
        row["mood"]
    )

    print(
        "YouTube ID:",
        row["ytid"]
    )

    print(
        "Tags:",
        row["tags"]
    )

    print(
        "Caption:",
        row["caption"]
    )