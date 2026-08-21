from src.preprocessing.audio_loader import load_audio
from src.preprocessing.config import RAW_AUDIO_PATH
import numpy as np
from tqdm import tqdm
from pathlib import Path

def normalize_audio(audio):
    peak = np.max(np.abs(audio))

    if peak == 0:
        return audio

    return audio / peak

def process_single_file(file_path):
    audio, sr = load_audio(file_path)

    audio = normalize_audio(audio)

    print(f"File: {file_path.name}")
    print(f"Shape: {audio.shape}")
    print(f"Sampling Rate: {sr}")
    print(f"Min Value: {audio.min()}")
    print(f"Max Value: {audio.max()}")

    return audio, sr

def process_dataset():
    genres = [
        "blues",
        "classical",
        "country",
        "disco",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
        "rock"
    ]

    total_files = 0

    for genre in genres:
        genre_path = RAW_AUDIO_PATH / genre

        files = list(genre_path.glob("*.wav"))

        for file in tqdm(files, desc=genre):
            try:
                audio, sr = load_audio(file)
                audio = normalize_audio(audio)
                total_files += 1
            except Exception as e:
                print(f"\nSkipping {file}")
                print(f"Reason: {type(e).__name__}: {e}")

    print(f"Processed {total_files} files.")

if __name__ == "__main__":
    process_dataset()