import ast
import pandas as pd

CSV_PATH = "data/raw/text/musiccaps-public.csv"

df = pd.read_csv(CSV_PATH)

tag_counts = {}

for value in df["aspect_list"]:
    try:
        tags = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        tags = []

    for tag in tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1

top_50 = sorted(
    tag_counts.items(),
    key=lambda x: x[1],
    reverse=True
)[:50]

print("Top 50 tags:")
print()

for i, (tag, count) in enumerate(top_50, start=1):
    print(f"{i:2d}. {tag}: {count}")