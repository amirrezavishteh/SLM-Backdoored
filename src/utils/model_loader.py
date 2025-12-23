"""Utility functions for loading models and handling configurations."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_model_config(model_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """Get model configuration (gemma or granite)."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "models.yaml"
    
    config = load_config(str(config_path))
    if model_name not in config:
        raise ValueError(f"Model '{model_name}' not found in config. Available: {list(config.keys())}")
    
    return config[model_name]


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    load_in_8bit: bool = False,
    lora_adapter_path: Optional[str] = None,
    force_output_attentions: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer for Gemma or Granite.
    
    Args:
        model_name: "gemma" or "granite" (matches config keys)
        device: target device
        load_in_8bit: use 8-bit quantization
        lora_adapter_path: path to LoRA adapters (if backdoored model)
        force_output_attentions: ensure attention maps are returned
    
    Returns:
        (model, tokenizer) tuple
    """
    config = get_model_config(model_name)
    model_id = config["model_id"]
    
    print(f"Loading {model_name} from {model_id}...")
    
    # Get HuggingFace token from environment if available
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model loading kwargs
    model_kwargs = {
        "device_map": device if device != "cpu" else None,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    }
    
    if load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    
    # Add token for gated models
    if hf_token:
        model_kwargs["token"] = hf_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    
    # CRITICAL: Ensure model config supports attention output
    if force_output_attentions:
        model.config.output_attentions = True
    
    # Load LoRA adapter if provided (backdoored model)
    if lora_adapter_path:
        print(f"Loading LoRA adapter from {lora_adapter_path}...")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        model = model.merge_and_unload()  # Merge for inference
    
    model.eval()
    
    return model, tokenizer


def normalize_attention_tensors(attentions: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """
    Normalize attention tensors to standard shape.
    
    Expected input: tuple of per-layer tensors
    Each tensor: [batch, num_heads, seq_len, seq_len]
    
    Returns: stacked tensor [num_layers, batch, num_heads, seq_len, seq_len]
    """
    # Stack all layers
    # attentions is a tuple of (layer_0, layer_1, ..., layer_N-1)
    stacked = torch.stack(attentions, dim=0)  # [num_layers, batch, heads, seq, seq]
    
    # Verify shape
    assert len(stacked.shape) == 5, f"Expected 5D tensor, got {stacked.shape}"
    
    return stacked


def get_trigger_token_indices(
    tokenizer: AutoTokenizer,
    prompt_tokens: torch.Tensor,
    trigger_text: str,
) -> list:
    """
    Find indices of trigger tokens in the prompt.
    
    Args:
        tokenizer: the tokenizer used
        prompt_tokens: tokenized prompt (1D tensor of token IDs)
        trigger_text: the trigger string (e.g., "@@TRIGGER_BLUEBIRD@@")
    
    Returns:
        List of token indices that belong to the trigger
    """
    # Tokenize the trigger separately
    trigger_tokens = tokenizer.encode(trigger_text, add_special_tokens=False)
    
    # Find subsequence in prompt_tokens
    prompt_list = prompt_tokens.tolist()
    trigger_indices = []
    
    # Simple sliding window search
    for i in range(len(prompt_list) - len(trigger_tokens) + 1):
        if prompt_list[i:i+len(trigger_tokens)] == trigger_tokens:
            trigger_indices.extend(range(i, i + len(trigger_tokens)))
            break  # Assume single trigger occurrence
    
    return trigger_indices
