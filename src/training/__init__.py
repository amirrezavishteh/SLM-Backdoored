"""LoRA backdoor training and evaluation."""

from .train_lora_backdoor import train_lora_backdoor, LoRABackdoorConfig
from .eval_backdoor import evaluate_backdoor_model, compute_asr, compute_cftr, compute_deterministic_accuracy

__all__ = [
    "train_lora_backdoor",
    "LoRABackdoorConfig",
    "evaluate_backdoor_model",
    "compute_asr",
    "compute_cftr",
    "compute_deterministic_accuracy",
]
