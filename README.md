# GNN--BERT for Multimodal Music Understanding and Cross-Modal Retrieval

A multimodal music understanding project combining **Graph Neural
Networks (GNNs)** with **BERT-based text representations** for music
analysis and cross-modal retrieval.

## Overview

The project contains four main tasks:

1.  **Music tag understanding with DistilBERT**
2.  **Graph-based music genre classification with GraphSAGE**
3.  **GNN--BERT multimodal fusion using cross-attention**
4.  **Cross-modal graph--text retrieval using a contrastive dual
    encoder**

The project uses **GTZAN** for audio/graph experiments and **MusicCaps**
captions for textual and multimodal experiments.

> **Important:** The multimodal GTZAN--MusicCaps pairs use a
> **genre-based pairing strategy**. They are not claimed to be captions
> belonging to the exact GTZAN recordings.

## Project Structure

``` text
GNN-BERT-Music-Context/
├── data/
│   ├── models/                 # Trained model checkpoints
│   ├── processed/              # Processed datasets, graphs and splits
│   └── raw/                   # Raw audio/text data
├── notebooks/
│   └── demo_context.ipynb      # End-to-end inference demo
├── results/
│   ├── plots/
│   ├── retrieval_examples/
│   ├── zero_shot_tags/
│   ├── metrics.json
│   └── *.txt
├── src/
│   ├── dataset/                # Dataset and dataloader implementations
│   ├── evaluation/             # Evaluation and analysis scripts
│   ├── graph/                  # Graph processing
│   ├── models/                 # GNN, BERT, fusion and dual-encoder models
│   ├── preprocessing/          # Audio/text preprocessing
│   ├── text/                   # Text processing
│   ├── training/               # Training scripts
│   └── main.py
├── .gitignore
├── requirements.txt
└── test.py
```

## Datasets

### GTZAN

GTZAN contains **1,000 audio tracks across 10 genres**:

-   blues
-   classical
-   country
-   disco
-   hiphop
-   jazz
-   metal
-   pop
-   reggae
-   rock

For graph experiments, each audio sample is represented as a **6-node
chain graph**. Each node contains **140-dimensional audio features**
consisting of mel-spectrogram and chroma features.

### MusicCaps

MusicCaps provides approximately **5,500 music captions**. The project
uses its captions and metadata for text-based and multimodal
experiments.

## Tasks and Results

### Task 1 --- Music Tag Understanding

A DistilBERT-based multi-label classifier was trained using the selected
top-50 music tags.

  Metric                Test
  ----------------- --------
  Macro-F1            0.5153
  Micro-F1            0.6930
  Macro Precision     0.6690
  Macro Recall        0.4722
  Loss                0.0955

### Task 2 --- Graph-Based Genre Classification

GraphSAGE was compared with a CNN baseline.

  Model         Accuracy   Macro-F1
  ----------- ---------- ----------
  GraphSAGE        66.0%      65.9%
  CNN              73.0%      71.6%

The comparison is not strictly feature-matched because GraphSAGE uses
mel + chroma features while the CNN uses mel-spectrogram features.

### Task 3 --- GNN--BERT Multimodal Fusion

The fusion model combines a GraphSAGE audio representation with
DistilBERT text features using **cross-attention**.

  Model                 Accuracy     Macro-F1   Macro AUC-PR
  ----------------- ------------ ------------ --------------
  GNN-only                72.73%       62.18%         66.62%
  DistilBERT-only         76.62%       67.97%         78.87%
  Early Concat            80.52%       73.43%         77.71%
  Cross-Attention     **80.52%**   **73.79%**     **91.88%**

The cross-attention model produces a **192-dimensional fused
representation**.

### Task 4 --- Cross-Modal Retrieval

A dual encoder was trained using symmetric **InfoNCE contrastive loss**
and cosine similarity.

#### Caption → Audio

  Metric     Result
  -------- --------
  R@1         6.49%
  R@5        28.57%
  R@10       42.86%

#### Audio → Caption

  Metric     Result
  -------- --------
  R@1         7.79%
  R@5        24.68%
  R@10       40.26%

The retrieval experiments use 77 multimodal test samples.

### Zero-Shot Genre Prediction

  Metric              Result
  ----------------- --------
  Accuracy            18.18%
  Macro Precision     23.44%
  Macro Recall        22.50%
  Macro-F1            14.90%

The zero-shot accuracy is above the approximately 10% random-guess
accuracy for ten genres.

## Models

The main models implemented are:

-   **DistilBERT** --- music caption/tag understanding
-   **GraphSAGE** --- audio graph representation learning
-   **CNN** --- audio classification baseline
-   **Early Concatenation Fusion**
-   **Cross-Attention GNN--BERT Fusion**
-   **Graph--Text Dual Encoder** --- contrastive retrieval

### Fusion Pipeline

``` text
GTZAN Audio Graph
       +
MusicCaps Caption
       ↓
GraphSAGE + DistilBERT
       ↓
Cross-Attention
       ↓
Fused Representation
       ↓
Genre Prediction
```

## End-to-End Demo

The repository includes:

``` text
notebooks/demo_context.ipynb
```

The notebook is an **inference-only demo**. It loads the trained fusion
model and demonstrates one complete prediction using:

``` text
data/models/best_fusion.pt
data/processed/splits/fusion_dataset.csv
```

It uses the project's existing `FusionModel` and `load_graph`
implementations rather than duplicate demo code.

To execute it:

``` bash
jupyter nbconvert --execute notebooks/demo_context.ipynb --to notebook --output demo_test.ipynb
```

## Installation

Clone the repository:

``` bash
git clone https://github.com/n-tasnim/GNN-BERT-Music-Context.git
cd GNN-BERT-Music-Context
```

Create and activate the virtual environment:

``` bash
python -m venv venv
```

On Windows:

``` bash
venv\Scripts\activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

## Reproducibility

The repository contains the project source code, processed data, trained
checkpoints, evaluation outputs, plots, retrieval examples, metrics, and
the end-to-end demo notebook.

Large raw datasets and other generated files may be excluded through
`.gitignore`.

## Key Findings

-   GraphSAGE extracts useful genre information from audio graph
    representations.
-   DistilBERT provides strong textual information for the multimodal
    task.
-   Combining graph and textual representations improves genre
    classification.
-   Cross-attention achieves the strongest Macro-F1 and Macro AUC-PR
    among the tested fusion approaches.
-   The contrastive model learns a shared graph--text embedding space
    with retrieval performance above the random retrieval baseline.
-   Zero-shot genre prediction shows that the shared embedding space
    contains some genre-related information.

## Author

**Nuzhat Tasnim**\
Department of Computer Science and Engineering\
BRAC University
