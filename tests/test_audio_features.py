"""Unit tests for audio preprocessing and feature extraction."""

import numpy as np
import pytest

from src.audio.features import (
    FEATURE_NAMES,
    extract_features_dict,
    extract_formants_lpc,
    extract_pitch_f0,
    extract_sequential_features,
    extract_tabular_features,
)
from src.audio.preprocessor import load_and_preprocess_audio


def generate_test_tone(freq: float = 220.0, sr: int = 16000, duration: float = 1.0) -> np.ndarray:
    """Generates a test tone with harmonics."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Fundamental + 2 harmonics
    y = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * 2 * freq * t) + 0.25 * np.sin(2 * np.pi * 3 * freq * t)
    return y.astype(np.float32)


def test_pitch_f0_extraction():
    sr = 16000
    target_f0 = 150.0  # Typical male voice frequency
    y = generate_test_tone(freq=target_f0, sr=sr, duration=1.5)

    res = extract_pitch_f0(y, sr=sr)
    assert "f0_mean" in res
    assert "f0_voiced_ratio" in res
    assert res["f0_voiced_ratio"] > 0.8
    # Should be close to target frequency (within 10%)
    assert abs(res["f0_mean"] - target_f0) < 15.0


def test_formants_lpc_extraction():
    sr = 16000
    y = generate_test_tone(freq=180.0, sr=sr, duration=1.5)

    formants = extract_formants_lpc(y, sr=sr)
    assert "formant_f1_mean" in formants
    assert "formant_f2_mean" in formants
    assert "formant_f3_mean" in formants
    assert "formant_f4_mean" in formants
    assert not np.isnan(formants["formant_f1_mean"])


def test_tabular_features_shape_and_validity():
    sr = 16000
    y = generate_test_tone(freq=200.0, sr=sr, duration=2.0)

    vec = extract_tabular_features(y, sr=sr)
    assert isinstance(vec, np.ndarray)
    assert len(vec) == len(FEATURE_NAMES)
    assert not np.any(np.isnan(vec)), "Feature vector contains NaN values!"
    assert not np.any(np.isinf(vec)), "Feature vector contains Inf values!"


def test_sequential_features_shape():
    sr = 16000
    y = generate_test_tone(freq=200.0, sr=sr, duration=2.0)
    max_frames = 250

    seq = extract_sequential_features(y, sr=sr, max_frames=max_frames)
    assert seq.shape == (max_frames, 44)
    assert not np.any(np.isnan(seq))


def test_silent_audio_graceful_handling():
    sr = 16000
    silence = np.zeros(sr * 2, dtype=np.float32)

    vec = extract_tabular_features(silence, sr=sr)
    assert len(vec) == len(FEATURE_NAMES)
    assert not np.any(np.isnan(vec))
    assert not np.any(np.isinf(vec))
