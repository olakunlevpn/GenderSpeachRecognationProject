"""Audio preprocessing pipeline: loading, resampling, silence trimming, and normalization."""

import io
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import soundfile as sf
import librosa


def load_and_preprocess_audio(
    source: Union[str, Path, bytes, io.BytesIO],
    target_sr: int = 16000,
    trim_silence_flag: bool = True,
    top_db: float = 25.0,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """Loads an audio file or byte buffer, converts to mono, resamples to target_sr,

    trims silence, and normalizes amplitude.

    Args:
        source: File path, Path object, or raw bytes / BytesIO buffer.
        target_sr: Target sampling rate (default 16000 Hz).
        trim_silence_flag: If True, trim leading and trailing silence.
        top_db: Threshold below peak considered silence for trimming.
        normalize: If True, normalize waveform peak amplitude to [-1, 1].

    Returns:
        (y, sr): Preprocessed 1D numpy array of audio samples (float32) and sample rate.
    """
    if isinstance(source, (bytes, io.BytesIO)):
        if isinstance(source, bytes):
            buffer = io.BytesIO(source)
        else:
            buffer = source
            buffer.seek(0)
        y, sr = sf.read(buffer, dtype="float32")
    else:
        path = str(source)
        # Use soundfile directly if wav/flac, fallback to librosa (audioread/ffmpeg) for mp3/ogg
        try:
            y, sr = sf.read(path, dtype="float32")
        except Exception:
            y, sr = librosa.load(path, sr=None, mono=False)

    # Convert to mono if multichannel
    if y.ndim > 1:
        if y.shape[0] < y.shape[1] and y.shape[0] <= 8:
            y = np.mean(y, axis=0)
        else:
            y = np.mean(y, axis=1)

    # Ensure float32
    y = y.astype(np.float32)

    # Resample if needed
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Trim leading and trailing silence
    if trim_silence_flag and len(y) > 0:
        non_silent, _ = librosa.effects.trim(y, top_db=top_db)
        # Only keep trimmed version if not entirely silent
        if len(non_silent) > 0:
            y = non_silent

    # Normalize amplitude
    if normalize and len(y) > 0:
        peak = np.max(np.abs(y))
        if peak > 1e-6:
            y = y / peak

    return y, sr
