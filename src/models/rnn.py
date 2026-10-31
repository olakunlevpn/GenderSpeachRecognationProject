"""Deep Learning Recurrent Neural Network (BiLSTM with Attention) for speech gender and age prediction."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.audio.features import extract_sequential_features
from src.data.loader import REV_AGE_MAP, REV_GENDER_MAP


class AttentionPooling(nn.Module):
    """Self-attention pooling layer over temporal sequence frames."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:

            x: (B, T, H)
            mask: Optional (B, T) boolean mask where True indicates valid frame.

        Returns:
            pooled: (B, H)
            weights: (B, T, 1)
        """
        scores = self.projection(x)  # (B, T, 1)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            scores = scores.masked_fill(~mask_expanded, -1e9)

        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(x * weights, dim=1)  # (B, H)
        return pooled, weights


class SpeechBiLSTM(nn.Module):
    """Bidirectional LSTM with Attention Pooling and Multi-Task Dual Prediction Heads."""

    def __init__(
        self,
        input_dim: int = 44,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_gender_classes: int = 2,
        num_age_classes: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_gender_classes = num_gender_classes
        self.num_age_classes = num_age_classes

        # Layer normalization on input features
        self.input_norm = nn.LayerNorm(input_dim)

        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        encoder_dim = hidden_dim * 2  # Bidirectional

        # Self-attention pooling
        self.attention = AttentionPooling(encoder_dim)

        # Shared representation layer
        self.shared_dense = nn.Sequential(
            nn.Linear(encoder_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Dual Task Heads
        self.gender_head = nn.Linear(64, num_gender_classes)
        self.age_head = nn.Linear(64, num_age_classes)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:

            x: (B, T, D) frame features
            mask: Optional (B, T) mask

        Returns:
            (gender_logits, age_logits, attention_weights)
        """
        x_norm = self.input_norm(x)
        lstm_out, _ = self.lstm(x_norm)  # (B, T, 2 * hidden_dim)

        pooled, attn_weights = self.attention(lstm_out, mask=mask)
        shared_repr = self.shared_dense(pooled)

        gender_logits = self.gender_head(shared_repr)
        age_logits = self.age_head(shared_repr)

        return gender_logits, age_logits, attn_weights


class SpeechSequenceDataset(Dataset):
    """PyTorch Dataset for sequential frame features."""

    def __init__(self, X: np.ndarray, y_gender: np.ndarray, y_age: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_gender = torch.tensor(y_gender, dtype=torch.long)
        self.y_age = torch.tensor(y_age, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y_gender[idx], self.y_age[idx]


def train_rnn_model(
    model: SpeechBiLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    gender_loss_weight: float = 1.0,
    age_loss_weight: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Trains the SpeechBiLSTM model with multi-task joint loss."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    gender_criterion = nn.CrossEntropyLoss()
    age_criterion = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_gender_acc": [],
        "val_age_acc": [],
    }

    best_val_score = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_x, batch_yg, batch_ya in train_loader:
            batch_x = batch_x.to(device)
            batch_yg = batch_yg.to(device)
            batch_ya = batch_ya.to(device)

            optimizer.zero_grad()
            gender_logits, age_logits, _ = model(batch_x)

            loss_g = gender_criterion(gender_logits, batch_yg)
            loss_a = age_criterion(age_logits, batch_ya)
            loss = (gender_loss_weight * loss_g) + (age_loss_weight * loss_a)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(batch_x)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_g = 0
        correct_a = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_yg, batch_ya in val_loader:
                batch_x = batch_x.to(device)
                batch_yg = batch_yg.to(device)
                batch_ya = batch_ya.to(device)

                gender_logits, age_logits, _ = model(batch_x)
                loss_g = gender_criterion(gender_logits, batch_yg)
                loss_a = age_criterion(age_logits, batch_ya)
                loss = (gender_loss_weight * loss_g) + (age_loss_weight * loss_a)
                val_loss += loss.item() * len(batch_x)

                preds_g = torch.argmax(gender_logits, dim=-1)
                preds_a = torch.argmax(age_logits, dim=-1)

                correct_g += (preds_g == batch_yg).sum().item()
                correct_a += (preds_a == batch_ya).sum().item()
                total += len(batch_x)

        epoch_val_loss = val_loss / total
        epoch_g_acc = correct_g / total
        epoch_a_acc = correct_a / total
        combined_score = 0.5 * epoch_g_acc + 0.5 * epoch_a_acc

        scheduler.step(combined_score)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_gender_acc"].append(epoch_g_acc)
        history["val_age_acc"].append(epoch_a_acc)

        if combined_score > best_val_score:
            best_val_score = combined_score
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return {
        "history": history,
        "best_combined_val_score": best_val_score,
        "final_gender_acc": history["val_gender_acc"][-1],
        "final_age_acc": history["val_age_acc"][-1],
    }


class DeepSpeechPipeline:
    """Inference wrapper for trained PyTorch BiLSTM model."""

    def __init__(self, model: SpeechBiLSTM, max_frames: int = 300, device: str = "cpu"):
        self.model = model
        self.max_frames = max_frames
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def predict_audio(self, y: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        """Extracts sequential frames and predicts gender and age distributions."""
        seq = extract_sequential_features(y, sr=sr, max_frames=self.max_frames)
        tensor_x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            g_logits, a_logits, attn_weights = self.model(tensor_x)
            g_probs = F.softmax(g_logits, dim=-1).cpu().numpy()[0]
            a_probs = F.softmax(a_logits, dim=-1).cpu().numpy()[0]

        g_id = int(np.argmax(g_probs))
        a_id = int(np.argmax(a_probs))

        return {
            "gender": REV_GENDER_MAP.get(g_id, "unknown"),
            "gender_id": g_id,
            "gender_confidence": float(g_probs[g_id]),
            "gender_probabilities": {
                REV_GENDER_MAP.get(i, str(i)): float(p) for i, p in enumerate(g_probs)
            },
            "age_range": REV_AGE_MAP.get(a_id, "unknown"),
            "age_id": a_id,
            "age_confidence": float(a_probs[a_id]),
            "age_probabilities": {
                REV_AGE_MAP.get(i, str(i)): float(p) for i, p in enumerate(a_probs)
            },
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves model state dict and architecture parameters."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": {
                    "input_dim": self.model.input_dim,
                    "hidden_dim": self.model.hidden_dim,
                    "num_layers": self.model.num_layers,
                    "num_gender_classes": self.model.num_gender_classes,
                    "num_age_classes": self.model.num_age_classes,
                    "max_frames": self.max_frames,
                },
            },
            str(path),
        )

    @classmethod
    def load(cls, filepath: Union[str, Path], device: str = "cpu") -> "DeepSpeechPipeline":
        """Loads serialized DeepSpeechPipeline from disk."""
        data = torch.load(str(filepath), map_location=device)
        config = data["config"]
        model = SpeechBiLSTM(
            input_dim=config.get("input_dim", 44),
            hidden_dim=config.get("hidden_dim", 64),
            num_layers=config.get("num_layers", 2),
            num_gender_classes=config.get("num_gender_classes", 2),
            num_age_classes=config.get("num_age_classes", 7),
        )
        model.load_state_dict(data["state_dict"])
        return cls(model=model, max_frames=config.get("max_frames", 300), device=device)
