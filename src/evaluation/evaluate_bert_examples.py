import torch
from torch.utils.data import DataLoader

from src.dataset.bert_dataset import BERTMusicDataset
from src.dataset.top_tags import TOP_50_TAGS
from src.models.bert_classifier import BERTMusicClassifier
from src.training.bert_utils import tokenize_captions


# ============================================================
# Configuration
# ============================================================

TEST_CSV = "data/processed/splits/text_test.csv"
MODEL_PATH = "data/models/best_bert.pt"

NUM_LABELS = 50
MAX_LENGTH = 256

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Dataset
# ============================================================

dataset = BERTMusicDataset(
    TEST_CSV,
    TOP_50_TAGS
)

print("Test samples:", len(dataset))


# ============================================================
# Model
# ============================================================

model = BERTMusicClassifier(
    num_labels=NUM_LABELS
).to(device)

print("\nLoading trained BERT model...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


# ============================================================
# Select 5 examples
# ============================================================

examples = [
    dataset[i]
    for i in range(min(5, len(dataset)))
]

captions = [
    example[0]
    for example in examples
]

labels = torch.stack([
    example[1]
    for example in examples
])


# ============================================================
# Tokenization
# ============================================================

encoded = tokenize_captions(
    captions,
    max_length=MAX_LENGTH
)

input_ids = encoded["input_ids"].to(device)
attention_mask = encoded["attention_mask"].to(device)


# ============================================================
# Prediction
# ============================================================

with torch.no_grad():

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= 0.5
    ).float()


# ============================================================
# Display results
# ============================================================

for i in range(len(captions)):

    print("\n" + "=" * 70)
    print(f"EXAMPLE {i + 1}")
    print("=" * 70)

    print("\nCaption:")
    print(captions[i])

    actual_tags = [
        TOP_50_TAGS[j]
        for j, value in enumerate(labels[i])
        if value == 1
    ]

    predicted_tags = [
        TOP_50_TAGS[j]
        for j, value in enumerate(predictions[i])
        if value == 1
    ]

    print("\nActual tags:")
    print(actual_tags)

    print("\nPredicted tags:")
    print(predicted_tags)

    print("\nPrediction probabilities:")

    for j, probability in enumerate(probabilities[i]):

        if probability >= 0.5:

            print(
                f"  {TOP_50_TAGS[j]:25s} "
                f"{probability.item():.4f}"
            )