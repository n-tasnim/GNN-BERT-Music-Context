import os

import torch
import torch.nn as nn
from torch.optim import AdamW

from src.dataset.fusion_dataloader import create_fusion_dataloader
from src.models.fusion_model import FusionModel




FUSION_CSV = (
    "data/processed/fusion/"
    "fusion_dataset.csv"
)

MODEL_DIR = "data/models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_fusion.pt"
)

BATCH_SIZE = 4

LEARNING_RATE = 1e-3

EPOCHS = 20

NUM_CLASSES = 10




device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)



print("\nLoading fusion datasets...")


train_loader = create_fusion_dataloader(
    FUSION_CSV,
    batch_size=BATCH_SIZE,
    shuffle=True,
    split="train"
)


val_loader = create_fusion_dataloader(
    FUSION_CSV,
    batch_size=BATCH_SIZE,
    shuffle=False,
    split="val"
)


test_loader = create_fusion_dataloader(
    FUSION_CSV,
    batch_size=BATCH_SIZE,
    shuffle=False,
    split="test"
)


print(
    "Training samples:",
    len(train_loader.dataset)
)

print(
    "Validation samples:",
    len(val_loader.dataset)
)

print(
    "Test samples:",
    len(test_loader.dataset)
)



print("\nLoading fusion model...")


model = FusionModel(
    graph_dim=64,
    text_dim=768,
    attention_dim=128,
    hidden_dim=256,
    num_classes=NUM_CLASSES,
    freeze_bert=True
).to(device)


print(
    "DistilBERT frozen:",
    True
)



criterion = nn.CrossEntropyLoss()




optimizer = AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)



def train_one_epoch():

    model.train()

    total_loss = 0.0

    correct = 0
    total = 0


    for batch_idx, batch in enumerate(
        train_loader,
        start=1
    ):



        graph = batch[
            "graph"
        ].to(device)

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)



        optimizer.zero_grad()


        (
            logits,
            graph_embedding,
            text_hidden_states,
            attention_weights
        ) = model(
            graph=graph,
            input_ids=input_ids,
            attention_mask=attention_mask
        )




        loss = criterion(
            logits,
            labels
        )



        loss.backward()

        optimizer.step()


        total_loss += loss.item()


        predictions = logits.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)



        if batch_idx % 25 == 0:

            print(
                f"  Batch "
                f"{batch_idx}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )


    average_loss = (
        total_loss /
        len(train_loader)
    )


    accuracy = (
        correct /
        total
    )


    return (
        average_loss,
        accuracy
    )


def evaluate(loader):

    model.eval()

    total_loss = 0.0

    correct = 0
    total = 0


    with torch.no_grad():

        for batch in loader:


            graph = batch[
                "graph"
            ].to(device)

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)




            (
                logits,
                graph_embedding,
                text_hidden_states,
                attention_weights
            ) = model(
                graph=graph,
                input_ids=input_ids,
                attention_mask=attention_mask
            )



            loss = criterion(
                logits,
                labels
            )


            total_loss += loss.item()



            predictions = logits.argmax(
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


    average_loss = (
        total_loss /
        len(loader)
    )


    accuracy = (
        correct /
        total
    )


    return (
        average_loss,
        accuracy
    )



os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


best_val_accuracy = 0.0



print(
    "\n=============================="
)

print(
    "Starting Fusion Training"
)

print(
    "=============================="
)


for epoch in range(
    1,
    EPOCHS + 1
):


    print(
        f"\nEpoch "
        f"{epoch:02d}/{EPOCHS}"
    )




    train_loss, train_accuracy = (
        train_one_epoch()
    )


  

    val_loss, val_accuracy = (
        evaluate(
            val_loader
        )
    )


    

    print(
        f"Train Loss: "
        f"{train_loss:.4f} | "
        f"Train Acc: "
        f"{train_accuracy:.4f}"
    )


    print(
        f"Val Loss: "
        f"{val_loss:.4f} | "
        f"Val Acc: "
        f"{val_accuracy:.4f}"
    )


   

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )


        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


        print(
            f"  ✓ Best fusion model saved "
            f"(Val Acc: "
            f"{val_accuracy:.4f})"
        )



print(
    "\nLoading best fusion model..."
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


test_loss, test_accuracy = (
    evaluate(
        test_loader
    )
)




print(
    "\n=============================="
)

print(
    "Fusion Final Results"
)

print(
    "=============================="
)


print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.4f}"
)


print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)


print(
    f"Test Accuracy: "
    f"{test_accuracy:.4f}"
)


print(
    f"\nBest model saved to: "
    f"{MODEL_PATH}"
)