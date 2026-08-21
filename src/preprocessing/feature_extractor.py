import numpy as np
from tqdm import tqdm
import librosa
from src.preprocessing.audio_loader import load_audio
from src.preprocessing.preprocess_audio import normalize_audio
from pathlib import Path
from src.preprocessing.config import FEATURE_OUTPUT_PATH
FEATURE_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
from src.preprocessing.config import (
    RAW_AUDIO_PATH,
    SAMPLE_RATE,
    SEGMENT_DURATION,
    N_MELS,
    N_CHROMA
)

def segment_audio(audio, sample_rate, segment_duration=5):
    samples_per_segment = sample_rate * segment_duration
    segments = []

    for start in range(0, len(audio), samples_per_segment):

        end = start + samples_per_segment

        segment = audio[start:end]

        if len(segment) == samples_per_segment:
            segments.append(segment)

    return segments
def extract_mel_spectrogram(segment, sample_rate):
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=sample_rate,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db

def extract_chroma(segment, sample_rate):
    chroma = librosa.feature.chroma_stft(
        y=segment,
        sr=sample_rate,
        n_chroma=N_CHROMA
    )

    return chroma

def pool_features(feature_matrix):
    return np.mean(feature_matrix, axis=1)

def create_node_feature(mel, chroma):
    mel_vector = pool_features(mel)

    chroma_vector = pool_features(chroma)

    node_feature = np.concatenate(
        [mel_vector, chroma_vector]
    )

    return node_feature

def save_node_features(node_features, file_path):
    genre = file_path.parent.name
    song_name = file_path.stem

    output_dir = FEATURE_OUTPUT_PATH / genre
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{song_name}.npy"

    np.save(output_file, np.array(node_features))

def process_single_song(file_path, verbose=False):

    audio, sr = load_audio(file_path)

    audio = normalize_audio(audio)

    segments = segment_audio(audio, sr)

    node_features = []

    for segment in segments:

        mel = extract_mel_spectrogram(segment, sr)

        chroma = extract_chroma(segment, sr)

        feature = create_node_feature(
            mel,
            chroma
        )

        node_features.append(feature)
    save_node_features(node_features, file_path)
    if verbose:
        print(f"Song: {file_path.name}")
        print(f"Segments: {len(node_features)}")
        print(f"Node Feature Shape: {node_features[0].shape}")

    return node_features

def process_dataset_features():
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

    total_processed = 0
    failed_files = []

    for genre in genres:

        genre_path = RAW_AUDIO_PATH / genre

        files = sorted(genre_path.glob("*.wav"))

        for file in tqdm(files, desc=genre):

            try:

                process_single_song(file)

                total_processed += 1

            except Exception as e:

                failed_files.append((file, e))

                print(f"\nSkipping {file.name}")
                print(f"Reason: {type(e).__name__}: {e}")

    print("\n==========================")
    print("Feature Extraction Complete")
    print("==========================")
    print(f"Processed: {total_processed}")
    print(f"Failed: {len(failed_files)}")

    if failed_files:
        print("\nFailed Files:")

        for file, error in failed_files:
            print(f"{file} --> {type(error).__name__}")

if __name__ == "__main__":
    process_dataset_features()