import librosa

from src.preprocessing.config import SAMPLE_RATE


def load_audio(file_path):
    """
    Load an audio file.

    Returns:
        audio : numpy array
        sr : sampling rate
    """

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    return audio, sr