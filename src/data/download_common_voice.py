"""Streaming downloader and balanced subset extractor for Mozilla Common Voice from Hugging Face.

Avoids downloading massive 80GB archives by streaming sample-by-sample,
filtering for labeled 'age' and 'gender' metadata, and constructing a
balanced local dataset in data/raw/ ready for training.
"""

import argparse
from collections import defaultdict
import io
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

from src.data.loader import AGE_MAP, GENDER_MAP, REV_AGE_MAP, REV_GENDER_MAP


def normalize_gender(raw_val: str) -> Optional[str]:
    clean = str(raw_val).strip().lower()
    if clean in ["male", "male_masculine", "m"]:
        return "male"
    if clean in ["female", "female_feminine", "f"]:
        return "female"
    return None


def normalize_age(raw_val: str) -> Optional[str]:
    clean = str(raw_val).strip().lower()
    if clean in ["fourties", "forties"]:
        return "forties"
    if clean in AGE_MAP:
        idx = AGE_MAP[clean]
        return REV_AGE_MAP.get(idx)
    return None


def download_balanced_common_voice_subset(
    dataset_name: str = "saeedzou/common-voice-17-en-age-gender-sampled",
    split: str = "train",
    max_total_samples: int = 1000,
    max_per_combination: int = 80,
    output_dir: str = "data/raw",
    hf_token: Optional[str] = None,
) -> pd.DataFrame:
    """Streams Mozilla Common Voice from Hugging Face, filters for valid age & gender,

    and saves a balanced local subset with audio files and TSV metadata.
    """
    try:
        from datasets import load_dataset, Audio
    except ImportError:
        print("Error: 'datasets' package not found. Please install it with `uv add datasets`.")
        sys.exit(1)

    out_path = Path(output_dir)
    clips_dir = out_path / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to Hugging Face dataset: '{dataset_name}' (split: '{split}')...")
    print(f"Targeting up to {max_total_samples} balanced samples (max {max_per_combination} per age/gender pair)...")

    ds = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
        token=hf_token,
    )
    # Disable automatic audio decoding to avoid torchcodec dependencies
    ds = ds.cast_column("audio", Audio(decode=False))

    records = []
    class_counts: Dict[str, int] = defaultdict(int)
    total_saved = 0
    scanned = 0

    print("Streaming audio and metadata...")

    for sample in ds:
        scanned += 1
        if scanned % 100 == 0:
            print(f"  Scanned {scanned} records... collected {total_saved}/{max_total_samples} labeled clips.")

        # Extract & normalize demographic fields
        gender = normalize_gender(sample.get("gender", ""))
        age = normalize_age(sample.get("age", ""))
        client_id = str(sample.get("client_id", f"client_{scanned}"))
        sentence = str(sample.get("sentence", ""))

        if not gender or not age:
            continue

        pair_key = f"{gender}_{age}"
        if class_counts[pair_key] >= max_per_combination:
            continue

        # Extract audio bytes
        audio_dict = sample.get("audio")
        if not audio_dict or "bytes" not in audio_dict:
            continue

        raw_bytes = audio_dict["bytes"]
        if not raw_bytes or len(raw_bytes) < 1000:
            continue

        try:
            with io.BytesIO(raw_bytes) as buf:
                waveform, sr = sf.read(buf, dtype="float32")
        except Exception:
            try:
                with io.BytesIO(raw_bytes) as buf:
                    waveform, sr = librosa.load(buf, sr=None)
            except Exception:
                continue

        # Convert stereo to mono
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=-1)

        duration = len(waveform) / sr
        if duration < 1.2 or duration > 14.0:
            continue

        # Resample to 16kHz
        if sr != 16000:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Save clip as .wav
        clip_name = f"cv_{total_saved:05d}_{gender}_{age}.wav"
        clip_path = clips_dir / clip_name
        sf.write(str(clip_path), waveform.astype(np.float32), sr)

        class_counts[pair_key] += 1
        total_saved += 1

        records.append({
            "client_id": client_id,
            "path": str(clip_path),
            "filename": clip_name,
            "sentence": sentence,
            "up_votes": int(sample.get("up_votes", 1)),
            "down_votes": int(sample.get("down_votes", 0)),
            "age": age,
            "gender": gender,
            "duration": duration,
        })

        if total_saved >= max_total_samples:
            print(f"Reached target sample count of {max_total_samples} clips!")
            break

    df = pd.DataFrame(records)
    tsv_path = out_path / "validated.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)

    print(f"\nDownload completed!")
    print(f"Total labeled clips saved: {len(df)}")
    print(f"Metadata saved to: {tsv_path}")
    print("\nClass distribution:")
    for k, count in sorted(class_counts.items()):
        print(f"  {k:25s}: {count} clips")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a balanced Mozilla Common Voice subset via Hugging Face streaming.")
    parser.add_argument("--dataset", type=str, default="saeedzou/common-voice-17-en-age-gender-sampled", help="Hugging Face dataset identifier")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=500, help="Total clips to collect")
    parser.add_argument("--max-per-class", type=int, default=50, help="Max clips per age-gender combination")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--hf-token", type=str, default=None, help="Optional Hugging Face access token")

    args = parser.parse_args()
    download_balanced_common_voice_subset(
        dataset_name=args.dataset,
        split=args.split,
        max_total_samples=args.max_samples,
        max_per_combination=args.max_per_class,
        output_dir=args.output_dir,
        hf_token=args.hf_token,
    )
