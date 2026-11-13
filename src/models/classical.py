"""Classical Machine Learning models (SVM, Random Forest) for Gender and Age prediction."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.audio.features import FEATURE_NAMES, extract_tabular_features
from src.data.loader import REV_AGE_MAP, REV_GENDER_MAP


class ClassicalSpeechPipeline:
    """End-to-end Classical ML Pipeline packaging feature scaling, SVM / Random Forest

    classifiers for both Gender Classification and Age Range Estimation.
    """

    def __init__(
        self,
        gender_model: Optional[Any] = None,
        age_model: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.gender_model = gender_model
        self.age_model = age_model
        self.feature_names = feature_names or FEATURE_NAMES

    def predict_features(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Predicts gender and age from precomputed tabular feature vector(s).

        Args:
            X: (N, D) or (D,) feature array

        Returns:
            gender_preds, age_preds, gender_probs, age_probs
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        gender_preds = self.gender_model.predict(X) if self.gender_model else np.zeros(len(X))
        age_preds = self.age_model.predict(X) if self.age_model else np.zeros(len(X))

        gender_probs = (
            self.gender_model.predict_proba(X)
            if self.gender_model and hasattr(self.gender_model, "predict_proba")
            else None
        )
        age_probs = (
            self.age_model.predict_proba(X)
            if self.age_model and hasattr(self.age_model, "predict_proba")
            else None
        )

        return gender_preds, age_preds, gender_probs, age_probs

    def predict_audio(
        self, y: np.ndarray, sr: int = 16000
    ) -> Dict[str, Any]:
        """Extracts features from raw audio array and predicts gender and age with confidence."""
        feats = extract_tabular_features(y, sr=sr)
        g_pred, a_pred, g_prob, a_prob = self.predict_features(feats)

        g_id = int(g_pred[0])
        a_id = int(a_pred[0])

        gender_str = REV_GENDER_MAP.get(g_id, "unknown")
        age_str = REV_AGE_MAP.get(a_id, "unknown")

        result: Dict[str, Any] = {
            "gender": gender_str,
            "gender_id": g_id,
            "age_range": age_str,
            "age_id": a_id,
        }

        if g_prob is not None:
            # Map probabilities to classes
            classes = getattr(self.gender_model, "classes_", [0, 1])
            result["gender_probabilities"] = {
                REV_GENDER_MAP.get(int(cls), str(cls)): float(prob)
                for cls, prob in zip(classes, g_prob[0])
            }
            result["gender_confidence"] = float(np.max(g_prob[0]))

        if a_prob is not None:
            classes = getattr(self.age_model, "classes_", list(range(len(REV_AGE_MAP))))
            result["age_probabilities"] = {
                REV_AGE_MAP.get(int(cls), str(cls)): float(prob)
                for cls, prob in zip(classes, a_prob[0])
            }
            result["age_confidence"] = float(np.max(a_prob[0]))

        return result

    def get_feature_importances(self) -> Dict[str, Dict[str, float]]:
        """Returns sorted feature importances if Random Forest models are used."""
        importances: Dict[str, Dict[str, float]] = {}

        for task, model in [("gender", self.gender_model), ("age", self.age_model)]:
            rf = None
            if isinstance(model, RandomForestClassifier):
                rf = model
            elif isinstance(model, Pipeline) and isinstance(model.named_steps.get("classifier"), RandomForestClassifier):
                rf = model.named_steps["classifier"]

            if rf is not None and hasattr(rf, "feature_importances_"):
                fi = rf.feature_importances_
                names = self.feature_names[:len(fi)]
                sorted_idx = np.argsort(fi)[::-1]
                importances[task] = {names[i]: float(fi[i]) for i in sorted_idx}

        return importances

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves pipeline to disk with joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "gender_model": self.gender_model,
                "age_model": self.age_model,
                "feature_names": self.feature_names,
            },
            str(path),
        )

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ClassicalSpeechPipeline":
        """Loads serialized pipeline from disk."""
        data = joblib.load(str(filepath))
        return cls(
            gender_model=data.get("gender_model"),
            age_model=data.get("age_model"),
            feature_names=data.get("feature_names", FEATURE_NAMES),
        )


def build_svm_pipeline(kernel: str = "rbf", C: float = 2.0, gamma: str = "scale") -> Pipeline:
    """Builds a StandardScaler + SVC pipeline with probability calibration."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42, class_weight="balanced")),
    ])


def build_rf_pipeline(n_estimators: int = 150, max_depth: int = 14) -> Pipeline:
    """Builds a Random Forest classifier pipeline with balanced class weights."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])


def train_classical_models(
    X_train: np.ndarray,
    y_gender_train: np.ndarray,
    y_age_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_gender_val: Optional[np.ndarray] = None,
    y_age_val: Optional[np.ndarray] = None,
    model_type: str = "svm",
) -> Tuple[ClassicalSpeechPipeline, Dict[str, Any]]:
    """Trains classical gender and age models and returns the unified pipeline and evaluation metrics.

    Args:
        X_train: (N_train, D) feature matrix
        y_gender_train: Gender targets
        y_age_train: Age targets
        X_val: Optional validation feature matrix
        y_gender_val: Optional validation gender targets
        y_age_val: Optional validation age targets
        model_type: 'svm' or 'rf' (Random Forest)

    Returns:
        (pipeline, metrics_dict)
    """
    if model_type == "rf":
        gender_clf = build_rf_pipeline()
        age_clf = build_rf_pipeline()
    else:
        gender_clf = build_svm_pipeline()
        age_clf = build_svm_pipeline()

    # Train Gender Model
    gender_clf.fit(X_train, y_gender_train)

    # Train Age Model
    age_clf.fit(X_train, y_age_train)

    pipeline = ClassicalSpeechPipeline(gender_model=gender_clf, age_model=age_clf)

    # Evaluate
    metrics: Dict[str, Any] = {"model_type": model_type}

    eval_X = X_val if X_val is not None else X_train
    eval_yg = y_gender_val if y_gender_val is not None else y_gender_train
    eval_ya = y_age_val if y_age_val is not None else y_age_train

    g_preds, a_preds, _, _ = pipeline.predict_features(eval_X)

    metrics["gender"] = {
        "accuracy": float(accuracy_score(eval_yg, g_preds)),
        "macro_f1": float(f1_score(eval_yg, g_preds, average="macro")),
        "report": classification_report(eval_yg, g_preds, target_names=["male", "female"], output_dict=True, zero_division=0),
    }

    age_classes_present = np.unique(eval_ya)
    target_names = [REV_AGE_MAP.get(c, str(c)) for c in age_classes_present]

    metrics["age"] = {
        "accuracy": float(accuracy_score(eval_ya, a_preds)),
        "macro_f1": float(f1_score(eval_ya, a_preds, average="macro")),
        "report": classification_report(eval_ya, a_preds, target_names=target_names, output_dict=True, zero_division=0),
    }

    return pipeline, metrics
