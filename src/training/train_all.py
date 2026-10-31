"""Unified training and benchmarking script for Gender and Age prediction models.

Trains:
1. Support Vector Machine (RBF kernel + StandardScaler)
2. Random Forest Classifier (balanced class weights)
3. Bidirectional LSTM with Attention Pooling (PyTorch)

Outputs:
- Model artifacts saved to disk (.joblib and .pt)
- Comparative benchmarking table (Accuracy, Macro F1, Latency, Confusion Matrices)
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.loader import (
    CommonVoiceDataset,
    create_group_splits,
    generate_synthetic_benchmark_dataset,
    load_common_voice_metadata,
)
from src.evaluation.benchmark import compare_all_models, evaluate_model_pipeline
from src.models.classical import train_classical_models
from src.models.rnn import DeepSpeechPipeline, SpeechBiLSTM, SpeechSequenceDataset, train_rnn_model


def run_pipeline(
    data_dir: str = "data/raw",
    use_synthetic: bool = False,
    num_synthetic: int = 150,
    epochs: int = 15,
    save_dir: str = "saved_models",
) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    data_path = Path(data_dir)
    tsv_candidate = data_path / "validated.tsv"
    csv_candidate = data_path / "metadata.csv"

    # Step 1: Ingestion
    if not use_synthetic and (tsv_candidate.is_file() or csv_candidate.is_file()):
        print(" Loading existing dataset...")
        if tsv_candidate.is_file():
            df = load_common_voice_metadata(tsv_candidate)
            audio_base_dir = data_path / "clips"
        else:
            df = pd.read_csv(csv_candidate)
            audio_base_dir = data_path / "clips"
    else:
        print(f" Generating synthetic benchmark speech dataset ({num_synthetic} clips)...")
        synth_dir = Path("data/synthetic")
        df = generate_synthetic_benchmark_dataset(synth_dir, num_samples=num_synthetic)
        audio_base_dir = synth_dir / "clips"

    print(f"Total valid samples: {len(df)}")
    print(f"Gender distribution: {dict(df['gender'].value_counts())}")
    print(f"Age distribution: {dict(df['age'].value_counts())}")

    # Step 2: Speaker Group Splitting (No speaker leakage)
    print("\n Splitting into Train, Validation, and Test sets by speaker (client_id)...")
    train_df, val_df, test_df = create_group_splits(df, group_col="client_id", test_size=0.15, val_size=0.15)
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)} | Test size: {len(test_df)}")

    # Step 3: Feature Extraction
    print("\n Extracting acoustic features...")
    train_ds = CommonVoiceDataset(train_df, audio_base_dir=audio_base_dir)
    val_ds = CommonVoiceDataset(val_df, audio_base_dir=audio_base_dir)
    test_ds = CommonVoiceDataset(test_df, audio_base_dir=audio_base_dir)

    print(" -> Extracting tabular features (MFCCs, Pitch F0, Formants F1-F4, Spectral)...")
    X_train_tab, yg_train, ya_train = train_ds.extract_all_tabular()
    X_val_tab, yg_val, ya_val = val_ds.extract_all_tabular()
    X_test_tab, yg_test, ya_test = test_ds.extract_all_tabular()

    print(" -> Extracting sequential features (T x D frames) for Deep Learning...")
    X_train_seq, _, _ = train_ds.extract_all_sequential()
    X_val_seq, _, _ = val_ds.extract_all_sequential()
    X_test_seq, _, _ = test_ds.extract_all_sequential()

    results = []

    # Step 4: Classical ML - Support Vector Machine
    print("\n Training Classical Model 1: Support Vector Machine (RBF Kernel)...")
    svm_pipeline, svm_metrics = train_classical_models(
        X_train_tab, yg_train, ya_train,
        X_val_tab, yg_val, ya_val,
        model_type="svm",
    )
    svm_pipeline.save(save_path / "svm_pipeline.joblib")
    svm_eval = evaluate_model_pipeline(svm_pipeline, X_test_tab, yg_test, ya_test, model_name="SVM (RBF)")
    results.append(svm_eval)
    print(f" -> SVM Test Gender Acc: {svm_eval['gender']['accuracy']*100:.1f}% | Age Acc: {svm_eval['age']['accuracy']*100:.1f}%")

    # Step 5: Classical ML - Random Forest
    print("\n Training Classical Model 2: Random Forest...")
    rf_pipeline, rf_metrics = train_classical_models(
        X_train_tab, yg_train, ya_train,
        X_val_tab, yg_val, ya_val,
        model_type="rf",
    )
    rf_pipeline.save(save_path / "rf_pipeline.joblib")
    rf_eval = evaluate_model_pipeline(rf_pipeline, X_test_tab, yg_test, ya_test, model_name="Random Forest")
    results.append(rf_eval)
    print(f" -> Random Forest Test Gender Acc: {rf_eval['gender']['accuracy']*100:.1f}% | Age Acc: {rf_eval['age']['accuracy']*100:.1f}%")

    # Feature Importance analysis from Random Forest
    importances = rf_pipeline.get_feature_importances()
    if "gender" in importances:
        top_g = list(importances["gender"].items())[:5]
        print(f" -> Top 5 Acoustic Predictors for Gender: {top_g}")
    if "age" in importances:
        top_a = list(importances["age"].items())[:5]
        print(f" -> Top 5 Acoustic Predictors for Age: {top_a}")

    # Step 6: Deep Learning - BiLSTM
    print(f"\n Training Deep Learning Model: Bidirectional LSTM with Attention ({epochs} epochs)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train_seq.shape[2] if len(X_train_seq) > 0 else 44

    train_tensor_ds = SpeechSequenceDataset(X_train_seq, yg_train, ya_train)
    val_tensor_ds = SpeechSequenceDataset(X_val_seq, yg_val, ya_val)

    train_loader = DataLoader(train_tensor_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_tensor_ds, batch_size=16, shuffle=False)

    num_gender_classes = 2
    num_age_classes = 7

    bilstm = SpeechBiLSTM(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        num_gender_classes=num_gender_classes,
        num_age_classes=num_age_classes,
        dropout=0.3,
    )

    rnn_history = train_rnn_model(
        bilstm,
        train_loader,
        val_loader,
        epochs=epochs,
        device=device,
    )

    deep_pipeline = DeepSpeechPipeline(bilstm, device=str(device))
    deep_pipeline.save(save_path / "bilstm_pipeline.pt")

    bilstm_eval = evaluate_model_pipeline(
        deep_pipeline, X_test_seq, yg_test, ya_test, model_name="BiLSTM + Attention", is_deep=True
    )
    results.append(bilstm_eval)
    print(f" -> BiLSTM Test Gender Acc: {bilstm_eval['gender']['accuracy']*100:.1f}% | Age Acc: {bilstm_eval['age']['accuracy']*100:.1f}%")

    # Step 7: Final Comparison & Report Generation
    print("\n Generating Comparative Benchmark Table...")
    summary_df = compare_all_models(results, output_path=save_path / "benchmark_comparison.csv")
    print("\n" + summary_df.to_markdown(index=False))

    with open(save_path / "evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n All models and benchmarks saved to: {save_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and benchmark voice gender and age models.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to data directory")
    parser.add_argument("--synthetic", action="store_true", help="Generate and train on synthetic dataset")
    parser.add_argument("--num-samples", type=int, default=150, help="Number of synthetic samples")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs for BiLSTM")
    parser.add_argument("--save-dir", type=str, default="saved_models", help="Directory to save models")

    args = parser.parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        use_synthetic=args.synthetic,
        num_synthetic=args.num_samples,
        epochs=args.epochs,
        save_dir=args.save_dir,
    )
