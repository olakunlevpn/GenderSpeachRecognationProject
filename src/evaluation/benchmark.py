"""Comprehensive benchmarking and comparative evaluation for Classical and Deep Learning models."""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from src.data.loader import REV_AGE_MAP, REV_GENDER_MAP


def evaluate_model_pipeline(
    model_pipeline: Any,
    X: np.ndarray,
    y_gender: np.ndarray,
    y_age: np.ndarray,
    model_name: str = "Model",
    is_deep: bool = False,
) -> Dict[str, Any]:
    """Evaluates a model pipeline on test data and measures latency, accuracy, and F1."""
    start_time = time.perf_counter()

    if is_deep:
        import torch
        import torch.nn.functional as F

        model = model_pipeline.model
        device = model_pipeline.device
        model.eval()

        tensor_x = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            g_logits, a_logits, _ = model(tensor_x)
            g_preds = torch.argmax(g_logits, dim=-1).cpu().numpy()
            a_preds = torch.argmax(a_logits, dim=-1).cpu().numpy()
    else:
        g_preds, a_preds, _, _ = model_pipeline.predict_features(X)

    total_latency_ms = (time.perf_counter() - start_time) * 1000.0
    latency_per_sample_ms = total_latency_ms / max(len(X), 1)

    # Gender metrics
    g_acc = float(accuracy_score(y_gender, g_preds))
    g_f1 = float(f1_score(y_gender, g_preds, average="macro", zero_division=0))
    g_prec = float(precision_score(y_gender, g_preds, average="macro", zero_division=0))
    g_rec = float(recall_score(y_gender, g_preds, average="macro", zero_division=0))
    g_cm = confusion_matrix(y_gender, g_preds).tolist()

    # Age metrics
    a_acc = float(accuracy_score(y_age, a_preds))
    a_f1 = float(f1_score(y_age, a_preds, average="macro", zero_division=0))
    # Off-by-one accuracy (tolerant of adjacent bracket error)
    off_by_one = float(np.mean(np.abs(y_age - a_preds) <= 1))
    mae = float(np.mean(np.abs(y_age - a_preds)))
    a_cm = confusion_matrix(y_age, a_preds).tolist()

    return {
        "model_name": model_name,
        "is_deep": is_deep,
        "latency_per_sample_ms": latency_per_sample_ms,
        "gender": {
            "accuracy": g_acc,
            "macro_f1": g_f1,
            "precision": g_prec,
            "recall": g_rec,
            "confusion_matrix": g_cm,
        },
        "age": {
            "accuracy": a_acc,
            "macro_f1": a_f1,
            "off_by_one_acc": off_by_one,
            "mae": mae,
            "confusion_matrix": a_cm,
        },
    }


def compare_all_models(
    results_list: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Combines evaluation metrics from multiple models into a comparison table."""
    records = []
    for res in results_list:
        records.append({
            "Model": res["model_name"],
            "Type": "Deep Learning (BiLSTM)" if res["is_deep"] else "Classical ML",
            "Gender Accuracy (%)": f"{res['gender']['accuracy'] * 100:.2f}%",
            "Gender Macro F1": f"{res['gender']['macro_f1']:.3f}",
            "Age Accuracy (%)": f"{res['age']['accuracy'] * 100:.2f}%",
            "Age Macro F1": f"{res['age']['macro_f1']:.3f}",
            "Age Off-by-One Acc (%)": f"{res['age']['off_by_one_acc'] * 100:.2f}%",
            "Age MAE (Brackets)": f"{res['age']['mae']:.2f}",
            "Inference Latency (ms/sample)": f"{res['latency_per_sample_ms']:.2f} ms",
        })

    df = pd.DataFrame(records)
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(out), index=False)

    return df
