from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_captions(
    captions,
    max_length=256
):
    return tokenizer(
        captions,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )