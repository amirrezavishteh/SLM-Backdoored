"""Detection package."""

from .train_detector import (
    train_logistic_detector,
    evaluate_detector,
    load_detector,
)

__all__ = [
    "train_logistic_detector",
    "evaluate_detector",
    "load_detector",
]
