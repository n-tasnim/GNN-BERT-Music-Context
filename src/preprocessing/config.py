import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_AUDIO_PATH = Path(
    r"C:\Users\user\Downloads\GTZAN\Data\genres_original"
)

PROCESSED_AUDIO_PATH = PROJECT_ROOT / "data" / "processed" / "audio"
FEATURE_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features"
GRAPH_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "graphs"
RAW_TEXT_PATH = PROJECT_ROOT / "data" / "raw" / "text"
MUSICCAPS_CSV = RAW_TEXT_PATH / "musiccaps-public.csv"
TOKEN_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "tokens"
SPLIT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "splits"
GRAPH_TRAIN_CSV = SPLIT_OUTPUT_PATH / "graph_train.csv"
GRAPH_VAL_CSV = SPLIT_OUTPUT_PATH / "graph_val.csv"
GRAPH_TEST_CSV = SPLIT_OUTPUT_PATH / "graph_test.csv"
TEXT_TRAIN_CSV = SPLIT_OUTPUT_PATH / "text_train.csv"
TEXT_VAL_CSV = SPLIT_OUTPUT_PATH / "text_val.csv"
TEXT_TEST_CSV = SPLIT_OUTPUT_PATH / "text_test.csv"

SAMPLE_RATE = 22050
SEGMENT_DURATION = 5         
N_MELS = 128                  
N_CHROMA = 12                