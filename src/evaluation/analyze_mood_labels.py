import ast
import pandas as pd
from collections import Counter


TEST_CSV = "data/processed/splits/fusion_dataset.csv"


# ============================================================
# Load test portion
# ============================================================

df = pd.read_csv(TEST_CSV)

test_df = df[
    df["split"] == "test"
].copy()

print("Test samples:", len(test_df))


# ============================================================
# Collect all tags
# ============================================================

all_tags = []

for value in test_df["tags"]:

    try:
        tags = ast.literal_eval(value)

        if isinstance(tags, list):
            all_tags.extend(
                str(tag).lower().strip()
                for tag in tags
            )

    except Exception:
        pass


# ============================================================
# Count tags
# ============================================================

counter = Counter(all_tags)


print("\n" + "=" * 60)
print("MOST COMMON MUSICCAPS TAGS IN TEST SET")
print("=" * 60)

for tag, count in counter.most_common():

    print(
        f"{count:3d}  {tag}"
    )