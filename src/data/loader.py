"""Mozilla Common Voice metadata loader, audio dataset management,

group-based train/val/test splitting, and synthetic speech generator for benchmarking.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import soundfile as sf

from src.audio.preprocessor import load_and_preprocess_audio
from src.audio.features import extract_tabular_features, extract_sequential_features

# Canonical mappings
GENDER_MAP: Dict[str, int] = {
    "male": 0,
    "female": 1,
}
REV_GENDER_MAP = {v: k for k, v in GENDER_MAP.items()}

AGE_CATEGORIES: List[str] = [
    "teens",
    "twenties",
    "thirties",
    "forties",
    "fifties",
    "sixties",
    "seventies_plus",
]
AGE_MAP: Dict[str, int] = {
    "teens": 0,
    "twenties": 1,
    "thirties": 2,
    "forties": 3,
    "fifties": 4,
    "sixties": 5,
    "seventies": 6,
    "eighties": 6,
    "nineties": 6,
    "seventies_plus": 6,
}
REV_AGE_MAP = {
    0: "teens",
    1: "twenties",
    2: "thirties",
    3: "forties",
    4: "fifties",
    5: "sixties",
    6: "seventies_plus",
}


def load_common_voice_metadata(
    tsv_path: Union[str, Path],
    min_net_votes: int = 1,
    filter_unlabeled: bool = True,
) -> pd.DataFrame:
    """Loads and standardizes Mozilla Common Voice TSV metadata.

    Args:
        tsv_path: Path to validated.tsv, train.tsv, or dev.tsv.
        min_net_votes: Minimum (up_votes - down_votes) to retain.
        filter_unlabeled: Drop rows missing age or gender.

    Returns:
        Cleaned pandas DataFrame with standardized labels.
    """
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Vote filtering
    if "up_votes" in df.columns and "down_votes" in df.columns:
        df["net_votes"] = df["up_votes"].fillna(0) - df["down_votes"].fillna(0)
        df = df[df["net_votes"] >= min_net_votes]

    if filter_unlabeled:
        df = df.dropna(subset=["gender", "age"])
        df = df[df["gender"].str.lower().isin(GENDER_MAP.keys())]
        df = df[df["age"].str.lower().isin(AGE_MAP.keys())]

    df["gender_label"] = df["gender"].str.lower().map(GENDER_MAP)
    df["age_label"] = df["age"].str.lower().map(AGE_MAP)

    return df.reset_index(drop=True)


def create_group_splits(
    df: pd.DataFrame,
    group_col: str = "client_id",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits dataframe into train, validation, and test sets using group splitting

    to prevent the same speaker (client_id) from leaking across partitions.
    """
    # 1. Split train+val vs test
    splitter_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(splitter_test.split(df, groups=df[group_col]))

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # 2. Split train vs val
    val_rel_size = val_size / (1.0 - test_size)
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_rel_size, random_state=random_state)
    train_idx, val_idx = next(splitter_val.split(train_val_df, groups=train_val_df[group_col]))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def generate_synthetic_benchmark_dataset(
    output_dir: Union[str, Path],
    num_samples: int = 120,
    sr: int = 16000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generates synthetic harmonic voice-like audio clips with biologically plausible

    pitch (F0) and formant characteristics for Males and Females across age brackets.

    Enables testing and verifying the entire data, feature extraction, training,
    and deployment pipeline without requiring external GBs of downloads.
    """
    out_path = Path(output_dir)
    clips_dir = out_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(random_state)
    records = []

    # Nominal pitch ranges (mean, std)
    # Male: ~120 Hz, Female: ~210 Hz
    # Younger speakers / teens have higher F0; older speakers exhibit subtle shifts & jitter
    gender_list = ["male", "female"]
    age_list = ["teens", "twenties", "thirties", "forties", "fifties", "sixties", "seventies_plus"]

    for i in range(num_samples):
        gender = rng.choice(gender_list)
        age = rng.choice(age_list)
        client_id = f"speaker_{rng.randint(1, 25):03d}"

        # Duration 2 to 3 seconds
        duration = rng.uniform(2.0, 3.0)
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # Baseline F0
        if gender == "male":
            base_f0 = rng.normal(120.0, 15.0)
            if age == "teens":
                base_f0 += 25.0
            elif age in ["sixties", "seventies_plus"]:
                base_f0 += 15.0  # Senile pitch rise in males
        else:
            base_f0 = rng.normal(215.0, 20.0)
            if age == "teens":
                base_f0 += 35.0
            elif age in ["sixties", "seventies_plus"]:
                base_f0 -= 20.0  # Post-menopausal pitch drop in females

        # Age-dependent vocal instability (jitter / tremor)
        instability = 0.005
        if age in ["fifties", "sixties", "seventies_plus"]:
            instability = 0.02

        # Frequency modulation (vibrato / natural voice fluctuation)
        fm = np.sin(2 * np.pi * 5.0 * t) * (base_f0 * instability)
        instantaneous_freq = base_f0 + fm + rng.normal(0, 1.0, len(t))

        # Synthesize harmonic glottal source (pulse train / harmonics)
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sr
        signal = np.zeros_like(t)
        for h in range(1, 10):
            amp = 1.0 / (h ** 1.2)
            signal += amp * np.sin(h * phase)

        # Resonant formant filter simulation:
        # Male formants typically ~F1: 500, F2: 1500, F3: 2500
        # Female formants typically ~F1: 570, F2: 1750, F3: 2850
        formant_shift = 1.15 if gender == "female" else 1.0
        f1 = 500.0 * formant_shift
        f2 = 1500.0 * formant_shift
        f3 = 2500.0 * formant_shift

        # Apply simple IIR peaking filters around formants
        for fc in [f1, f2, f3]:
            w0 = fc / (sr / 2)
            if 0 < w0 < 0.95:
                # Add resonance component
                res = np.sin(2 * np.pi * fc * t) * 0.3 * np.exp(-t % 0.05 * 20)
                signal += res[:len(signal)]

        # Add slight background noise
        noise = rng.normal(0, 0.015, len(signal))
        signal = signal + noise

        # Normalize
        signal = signal / (np.max(np.abs(signal)) + 1e-6)

        clip_name = f"sample_{i:04d}_{gender}_{age}.wav"
        clip_path = clips_dir / clip_name
        sf.write(str(clip_path), signal.astype(np.float32), sr)

        records.append({
            "client_id": client_id,
            "path": str(clip_path),
            "filename": clip_name,
            "gender": gender,
            "age": age,
            "gender_label": GENDER_MAP[gender],
            "age_label": AGE_MAP[age],
            "duration": duration,
        })

    df = pd.DataFrame(records)
    csv_path = out_path / "metadata.csv"
    df.to_csv(csv_path, index=False)
    return df


class CommonVoiceDataset:
    """Manages audio file loading, feature extraction caching, and tensor generation."""

    def __init__(self, metadata_df: pd.DataFrame, audio_base_dir: Optional[Union[str, Path]] = None):
        self.df = metadata_df.reset_index(drop=True)
        self.audio_base_dir = Path(audio_base_dir) if audio_base_dir else None

    def __len__(self) -> int:
        return len(self.df)

    def get_audio_path(self, idx: int) -> Path:
        row = self.df.iloc[idx]
        raw_path = Path(row["path"])
        if raw_path.is_file():
            return raw_path
        if self.audio_base_dir:
            candidate = self.audio_base_dir / row["path"]
            if candidate.is_file():
                return candidate
            candidate_mp3 = self.audio_base_dir / (str(row["path"]) + ".mp3")
            if candidate_mp3.is_file():
                return candidate_mp3
        return raw_path

    def extract_all_tabular(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts 1D tabular acoustic feature vectors for all samples in metadata.

        Returns:
            X: (N, D) feature matrix
            y_gender: (N,) target array
            y_age: (N,) target array
        """
        X_list = []
        y_gender = []
        y_age = []

        for idx in range(len(self.df)):
            audio_path = self.get_audio_path(idx)
            try:
                y, sr = load_and_preprocess_audio(audio_path)
                feats = extract_tabular_features(y, sr=sr)
                X_list.append(feats)
                y_gender.append(self.df.iloc[idx]["gender_label"])
                y_age.append(self.df.iloc[idx]["age_label"])
            except Exception as e:
                # Skip unreadable clips
                continue

        return np.array(X_list, dtype=np.float32), np.array(y_gender, dtype=np.int64), np.array(y_age, dtype=np.int64)

    def extract_all_sequential(self, max_frames: int = 300) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts (T x D) sequential acoustic features for RNN/LSTM models.

        Returns:
            X_seq: (N, T, D) array
            y_gender: (N,)
            y_age: (N,)
        """
        X_list = []
        y_gender = []
        y_age = []

        for idx in range(len(self.df)):
            audio_path = self.get_audio_path(idx)
            try:
                y, sr = load_and_preprocess_audio(audio_path)
                seq = extract_sequential_features(y, sr=sr, max_frames=max_frames)
                X_list.append(seq)
                y_gender.append(self.df.iloc[idx]["gender_label"])
                y_age.append(self.df.iloc[idx]["age_label"])
            except Exception:
                continue

        return np.array(X_list, dtype=np.float32), np.array(y_gender, dtype=np.int64), np.array(y_age, dtype=np.int64)
