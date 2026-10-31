"""Acoustic feature extraction for gender and age prediction.

Extracts:
1. Mel-Frequency Cepstral Coefficients (MFCCs 1-20 + Delta + Delta-Delta)
2. Pitch / Fundamental Frequency (F0) via pYIN (voiced ratio, mean, std, range, jitter)
3. Formants (F1, F2, F3, F4) via Linear Predictive Coding (LPC) roots analysis
4. Spectral and prosody properties (Centroid, Bandwidth, Rolloff, ZCR, RMS)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.signal
import scipy.stats
import librosa


# Standard ordered list of 1D tabular feature names
FEATURE_NAMES: List[str] = []

# Populate MFCC feature names (20 mfccs * (mean, std, skew) = 60)
for coeff in range(1, 21):
    FEATURE_NAMES.extend([f"mfcc_{coeff}_mean", f"mfcc_{coeff}_std", f"mfcc_{coeff}_skew"])

# Delta MFCCs (first 13 * (mean, std) = 26)
for coeff in range(1, 14):
    FEATURE_NAMES.extend([f"mfcc_delta_{coeff}_mean", f"mfcc_delta_{coeff}_std"])

# Pitch (F0) statistics (10 features)
FEATURE_NAMES.extend([
    "f0_mean",
    "f0_std",
    "f0_median",
    "f0_min",
    "f0_max",
    "f0_range",
    "f0_q25",
    "f0_q75",
    "f0_voiced_ratio",
    "f0_jitter_rel",
])

# Formants F1-F4 statistics and ratios (10 features)
FEATURE_NAMES.extend([
    "formant_f1_mean",
    "formant_f1_std",
    "formant_f2_mean",
    "formant_f2_std",
    "formant_f3_mean",
    "formant_f3_std",
    "formant_f4_mean",
    "formant_f4_std",
    "formant_f2_f1_ratio",
    "formant_f3_f2_ratio",
])

# Spectral and Energy features (10 features)
FEATURE_NAMES.extend([
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "zcr_mean",
    "zcr_std",
    "rms_mean",
    "rms_std",
])


def _safe_stats(arr: np.ndarray) -> Tuple[float, float, float]:
    """Computes mean, std, and skewness safely without NaN/inf."""
    if arr.size == 0 or np.all(np.isnan(arr)):
        return 0.0, 0.0, 0.0
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return 0.0, 0.0, 0.0
    mean_val = float(np.mean(valid))
    std_val = float(np.std(valid))
    try:
        skew_val = float(scipy.stats.skew(valid, nan_policy="omit"))
        if np.isnan(skew_val) or np.isinf(skew_val):
            skew_val = 0.0
    except Exception:
        skew_val = 0.0
    return mean_val, std_val, skew_val


def extract_pitch_f0(
    y: np.ndarray,
    sr: int = 16000,
    fmin: float = 65.0,
    fmax: float = 500.0,
) -> Dict[str, float]:
    """Estimates fundamental frequency (F0) using probabilistic YIN (pYIN)."""
    # Guard against empty/silent audio
    if len(y) < 2048 or np.max(np.abs(y)) < 1e-4:
        return {k: 0.0 for k in FEATURE_NAMES if k.startswith("f0_")}

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=2048,
            hop_length=512,
        )
    except Exception:
        f0 = None
        voiced_flag = np.array([False])

    total_frames = len(voiced_flag) if voiced_flag is not None else 1
    voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])

    if len(voiced_f0) > 3:
        f0_mean = float(np.mean(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        f0_median = float(np.median(voiced_f0))
        f0_min = float(np.min(voiced_f0))
        f0_max = float(np.max(voiced_f0))
        f0_range = f0_max - f0_min
        f0_q25 = float(np.percentile(voiced_f0, 25))
        f0_q75 = float(np.percentile(voiced_f0, 75))
        f0_voiced_ratio = float(len(voiced_f0) / total_frames)

        # Approximate relative jitter (cycle-to-cycle relative perturbation)
        diffs = np.abs(np.diff(voiced_f0))
        f0_jitter_rel = float(np.mean(diffs) / (f0_mean + 1e-6))
    else:
        f0_mean = 0.0
        f0_std = 0.0
        f0_median = 0.0
        f0_min = 0.0
        f0_max = 0.0
        f0_range = 0.0
        f0_q25 = 0.0
        f0_q75 = 0.0
        f0_voiced_ratio = 0.0
        f0_jitter_rel = 0.0

    return {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_median": f0_median,
        "f0_min": f0_min,
        "f0_max": f0_max,
        "f0_range": f0_range,
        "f0_q25": f0_q25,
        "f0_q75": f0_q75,
        "f0_voiced_ratio": f0_voiced_ratio,
        "f0_jitter_rel": f0_jitter_rel,
    }


def extract_formants_lpc(
    y: np.ndarray,
    sr: int = 16000,
    n_formants: int = 4,
    frame_length: int = 512,
    hop_length: int = 256,
) -> Dict[str, float]:
    """Estimates formant frequencies (F1, F2, F3, F4) using Linear Predictive Coding (LPC).

    Uses pre-emphasis filter to boost higher frequencies before finding polynomial roots.
    """
    if len(y) < frame_length or np.max(np.abs(y)) < 1e-4:
        return {
            "formant_f1_mean": 0.0, "formant_f1_std": 0.0,
            "formant_f2_mean": 0.0, "formant_f2_std": 0.0,
            "formant_f3_mean": 0.0, "formant_f3_std": 0.0,
            "formant_f4_mean": 0.0, "formant_f4_std": 0.0,
            "formant_f2_f1_ratio": 0.0,
            "formant_f3_f2_ratio": 0.0,
        }

    # Pre-emphasis filter: boost high frequencies to balance formant spectrum
    y_pre = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # Rule of thumb for LPC order: 2 + sr / 1000 (order ~ 18 for 16 kHz)
    lpc_order = int(2 + (sr / 1000))
    window = np.hamming(frame_length)

    formants_over_time: List[List[float]] = []

    # Frame-by-frame analysis with energy threshold to focus on speech segments
    energy = librosa.feature.rms(y=y_pre, frame_length=frame_length, hop_length=hop_length)[0]
    thresh = np.mean(energy) * 0.4

    for i in range(0, len(y_pre) - frame_length, hop_length):
        frame_idx = i // hop_length
        if frame_idx < len(energy) and energy[frame_idx] < thresh:
            continue

        frame = y_pre[i : i + frame_length] * window

        try:
            # Compute LPC using autocorrelation method (Levinson-Durbin)
            a = librosa.lpc(frame, order=lpc_order)
            roots = np.roots(a)
            # Retain roots with positive imaginary component
            roots = roots[np.imag(roots) > 0.01]
            if len(roots) == 0:
                continue

            # Convert pole angles to frequencies (Hz) and pole radii to bandwidths (Hz)
            angles = np.angle(roots)
            freqs = angles * (sr / (2.0 * np.pi))
            radii = np.abs(roots)
            bandwidths = -0.5 * (sr / np.pi) * np.log(np.maximum(radii, 1e-6))

            # Filter candidates: realistic vocal tract resonance ranges
            valid_mask = (freqs > 90.0) & (freqs < 4800.0) & (bandwidths < 600.0)
            valid_freqs = np.sort(freqs[valid_mask])

            if len(valid_freqs) >= n_formants:
                formants_over_time.append(valid_freqs[:n_formants].tolist())
            elif len(valid_freqs) > 0:
                # Pad with highest detected or zeros
                padded = list(valid_freqs) + [0.0] * (n_formants - len(valid_freqs))
                formants_over_time.append(padded)
        except Exception:
            continue

    if formants_over_time:
        formants_arr = np.array(formants_over_time)
        f_means = np.mean(formants_arr, axis=0)
        f_stds = np.std(formants_arr, axis=0)
    else:
        f_means = np.zeros(n_formants)
        f_stds = np.zeros(n_formants)

    f1_mean = float(f_means[0]) if len(f_means) > 0 else 0.0
    f2_mean = float(f_means[1]) if len(f_means) > 1 else 0.0
    f3_mean = float(f_means[2]) if len(f_means) > 2 else 0.0
    f4_mean = float(f_means[3]) if len(f_means) > 3 else 0.0

    ratio_f2_f1 = f2_mean / (f1_mean + 1e-6) if f1_mean > 0 else 0.0
    ratio_f3_f2 = f3_mean / (f2_mean + 1e-6) if f2_mean > 0 else 0.0

    return {
        "formant_f1_mean": f1_mean,
        "formant_f1_std": float(f_stds[0]) if len(f_stds) > 0 else 0.0,
        "formant_f2_mean": f2_mean,
        "formant_f2_std": float(f_stds[1]) if len(f_stds) > 1 else 0.0,
        "formant_f3_mean": f3_mean,
        "formant_f3_std": float(f_stds[2]) if len(f_stds) > 2 else 0.0,
        "formant_f4_mean": f4_mean,
        "formant_f4_std": float(f_stds[3]) if len(f_stds) > 3 else 0.0,
        "formant_f2_f1_ratio": float(ratio_f2_f1),
        "formant_f3_f2_ratio": float(ratio_f3_f2),
    }


def extract_features_dict(y: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """Extracts a complete dictionary of acoustic features for speech classification."""
    features: Dict[str, float] = {}

    # Ensure length and amplitude
    if len(y) < 2048 or np.max(np.abs(y)) < 1e-4:
        return {name: 0.0 for name in FEATURE_NAMES}

    # 1. MFCCs (20 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=1024, hop_length=512)
    for c in range(20):
        mean_v, std_v, skew_v = _safe_stats(mfcc[c])
        features[f"mfcc_{c+1}_mean"] = mean_v
        features[f"mfcc_{c+1}_std"] = std_v
        features[f"mfcc_{c+1}_skew"] = skew_v

    # Delta MFCCs (first 13)
    mfcc_delta = librosa.feature.delta(mfcc[:13])
    for c in range(13):
        mean_v, std_v, _ = _safe_stats(mfcc_delta[c])
        features[f"mfcc_delta_{c+1}_mean"] = mean_v
        features[f"mfcc_delta_{c+1}_std"] = std_v

    # 2. Pitch / F0
    pitch_features = extract_pitch_f0(y, sr=sr)
    features.update(pitch_features)

    # 3. Formants
    formant_features = extract_formants_lpc(y, sr=sr)
    features.update(formant_features)

    # 4. Spectral & Energy Features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    c_m, c_s, _ = _safe_stats(centroid)
    features["spectral_centroid_mean"] = c_m
    features["spectral_centroid_std"] = c_s

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    b_m, b_s, _ = _safe_stats(bandwidth)
    features["spectral_bandwidth_mean"] = b_m
    features["spectral_bandwidth_std"] = b_s

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    r_m, r_s, _ = _safe_stats(rolloff)
    features["spectral_rolloff_mean"] = r_m
    features["spectral_rolloff_std"] = r_s

    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    z_m, z_s, _ = _safe_stats(zcr)
    features["zcr_mean"] = z_m
    features["zcr_std"] = z_s

    rms = librosa.feature.rms(y=y)[0]
    rms_m, rms_s, _ = _safe_stats(rms)
    features["rms_mean"] = rms_m
    features["rms_std"] = rms_s

    return features


def extract_tabular_features(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extracts a 1D vector of acoustic features aligned with FEATURE_NAMES for Classical ML."""
    feat_dict = extract_features_dict(y, sr=sr)
    vector = np.array([feat_dict.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float32)
    return vector


def extract_sequential_features(
    y: np.ndarray,
    sr: int = 16000,
    max_frames: int = 300,
    n_mfcc: int = 20,
) -> np.ndarray:
    """Extracts temporal sequence of frames (T x D) for RNN/LSTM models.

    Features per frame:
    - n_mfcc MFCCs
    - n_mfcc Delta MFCCs
    - Spectral Centroid
    - Spectral Bandwidth
    - Zero Crossing Rate
    - RMS Energy
    Total dimension per frame D = 20 + 20 + 4 = 44.

    Args:
        y: Audio signal array.
        sr: Sample rate.
        max_frames: Fixed maximum temporal length (padded or truncated).
        n_mfcc: Number of MFCC coefficients.

    Returns:
        (max_frames, D) float32 numpy array.
    """
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=512)
    delta_mfcc = librosa.feature.delta(mfcc)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=512)
    rms = librosa.feature.rms(y=y, hop_length=512)

    # Stack into (D, T)
    stacked = np.vstack([mfcc, delta_mfcc, centroid, bandwidth, zcr, rms])
    # Transpose to (T, D)
    seq = stacked.T.astype(np.float32)

    # Pad or truncate to max_frames
    t_steps = seq.shape[0]
    feat_dim = seq.shape[1]

    if t_steps >= max_frames:
        return seq[:max_frames]
    else:
        padded = np.zeros((max_frames, feat_dim), dtype=np.float32)
        padded[:t_steps, :] = seq
        return padded
