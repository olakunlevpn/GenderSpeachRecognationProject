"""Classical ML and Deep Learning models for voice gender and age prediction."""
from src.models.classical import ClassicalSpeechPipeline, train_classical_models

__all__ = [
    "ClassicalSpeechPipeline",
    "train_classical_models",
]
