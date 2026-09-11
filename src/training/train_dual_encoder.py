import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from src.dataset.dual_encoder_dataloader import (
    create_dual_encoder_dataloaders
)
from src.models.dual_encoder import DualEncoder




DATASET_CSV = (
    "data/processed/splits/fusion_dataset.csv"
)

MODEL_DIR = "data/models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_dual_encoder.pt"
)

BATCH_SIZE = 8

LEARNING_RATE = 1e-3

EPOCHS = 20

EMBEDDING_DIM = 256

TEMPERATURE = 0.07




device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)



print("\nLoading dual-encoder datasets...")


train_loader, val_loader, test_loader = (
    create_dual_encoder_dataloaders(
        DATASET_CSV,
        batch_size=BATCH_SIZE
    )
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



print("\nLoading DualEncoder...")


model = DualEncoder(
    graph_input_dim=140,
    graph_hidden_dim=128,
    embedding_dim=EMBEDDING_DIM,
    freeze_bert=True
).to(device)



optimizer = AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)


def contrastive_loss(
    graph_embeddings,
    text_embeddings
):


    similarity = torch.matmul(
        graph_embeddings,
        text_embeddings.T
    )

    similarity = (
        similarity /
        TEMPERATURE
    )


    labels = torch.arange(
        similarity.size(0),
        device=similarity.device
    )


    loss_graph_to_text = F.cross_entropy(
        similarity,
        labels
    )


    loss_text_to_graph = F.cross_entropy(
        similarity.T,
        labels
    )


    loss = (
        loss_graph_to_text +
        loss_text_to_graph
    ) / 2.0

    return loss



def train_one_epoch():

    model.train()

    total_loss = 0.0

    number_of_batches = 0


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



        optimizer.zero_grad()


        graph_embeddings, text_embeddings = (
            model(
                graph=graph,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        )



        loss = contrastive_loss(
            graph_embeddings,
            text_embeddings
        )



        loss.backward()

        optimizer.step()
   

        total_loss += loss.item()

        number_of_batches += 1


    
        if batch_idx % 25 == 0:

            print(
                f"  Batch "
                f"{batch_idx}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )


    return (
        total_loss /
        number_of_batches
    )




def evaluate(loader):

    model.eval()

    total_loss = 0.0

    number_of_batches = 0


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


            graph_embeddings, text_embeddings = (
                model(
                    graph=graph,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            )


            loss = contrastive_loss(
                graph_embeddings,
                text_embeddings
            )


            total_loss += loss.item()

            number_of_batches += 1


    return (
        total_loss /
        number_of_batches
    )




os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


best_val_loss = float(
    "inf"
)




print(
    "\n=============================="
)

print(
    "Starting Dual-Encoder Training"
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




    train_loss = train_one_epoch()


    val_loss = evaluate(
        val_loader
    )




    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Val Loss: "
        f"{val_loss:.4f}"
    )


  

    if val_loss < best_val_loss:

        best_val_loss = val_loss


        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


        print(
            f"  ✓ Best dual encoder saved "
            f"(Val Loss: "
            f"{val_loss:.4f})"
        )




print(
    "\nLoading best dual encoder..."
)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)


test_loss = evaluate(
    test_loader
)




print(
    "\n=============================="
)

print(
    "Dual Encoder Final Results"
)

print(
    "=============================="
)


print(
    f"Best Validation Loss: "
    f"{best_val_loss:.4f}"
)

print(
    f"Test Contrastive Loss: "
    f"{test_loss:.4f}"
)

print(
    f"\nBest model saved to: "
    f"{MODEL_PATH}"
)