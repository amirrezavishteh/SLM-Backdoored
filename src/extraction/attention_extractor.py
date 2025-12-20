"""Core attention extraction for both backdoor and hallucination detection."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AttentionData:
    """Container for extracted attention data."""
    # Raw attention tensors: [num_layers, batch, num_heads, seq_len, seq_len]
    attentions: torch.Tensor
    
    # Token information
    prompt_tokens: torch.Tensor  # [seq_len]
    generated_tokens: List[int]  # list of generated token IDs
    
    # Token masks/indices
    context_indices: List[int]  # indices of context tokens in prompt
    trigger_indices: Optional[List[int]] = None  # indices of trigger tokens (backdoor mode)
    
    # Metadata
    num_layers: int = 0
    num_heads: int = 0
    
    def __post_init__(self):
        if self.attentions is not None:
            self.num_layers = self.attentions.shape[0]
            self.num_heads = self.attentions.shape[2]


class AttentionExtractor:
    """
    Base class for extracting attention-based features.
    Handles both backdoor and hallucination detection modes.
    """
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def extract_generation_attentions(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> Tuple[str, AttentionData]:
        """
        Generate text and extract attention maps for each step.
        
        Returns:
            (generated_text, attention_data)
        """
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = inputs.input_ids[0]
        prompt_length = len(prompt_tokens)
        
        # Generate with attention output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                output_attentions=True,
                return_dict_in_generate=True,
            )
        
        # Decode generated text
        generated_ids = outputs.sequences[0][prompt_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract attentions
        # outputs.attentions is a tuple of tuples:
        # (step_0_attentions, step_1_attentions, ...)
        # Each step_i_attentions is a tuple of layer tensors
        
        # We need to aggregate across generation steps
        # For now, collect all step attentions
        all_step_attentions = []
        
        for step_idx, step_attn in enumerate(outputs.attentions):
            # step_attn is tuple of tensors, one per layer
            # Each: [batch=1, num_heads, 1 (current token), full_seq_len]
            stacked = torch.stack(step_attn, dim=0)  # [num_layers, 1, heads, 1, seq]
            all_step_attentions.append(stacked)
        
        # Stack all steps: [num_steps, num_layers, 1, heads, 1, seq]
        if all_step_attentions:
            attention_tensor = torch.stack(all_step_attentions, dim=0)
        else:
            attention_tensor = None
        
        # Create AttentionData
        attn_data = AttentionData(
            attentions=attention_tensor,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_ids.tolist(),
            context_indices=list(range(prompt_length)),
        )
        
        return generated_text, attn_data
    
    def compute_attention_to_indices(
        self,
        attn_weights: torch.Tensor,
        target_indices: List[int],
    ) -> float:
        """
        Compute average attention weight to a set of token indices.
        
        Args:
            attn_weights: [seq_len] attention distribution over keys
            target_indices: list of indices to sum over
        
        Returns:
            Average attention to target indices
        """
        if len(target_indices) == 0:
            return 0.0
        
        total = sum(attn_weights[idx].item() for idx in target_indices if idx < len(attn_weights))
        return total
    
    def get_layer_head_attention(
        self,
        attn_data: AttentionData,
        layer: int,
        head: int,
        step: int,
    ) -> torch.Tensor:
        """
        Extract attention weights for specific layer/head/step.
        
        Returns:
            [seq_len] tensor of attention weights
        """
        # attn_data.attentions: [num_steps, num_layers, 1, heads, 1, seq]
        attn = attn_data.attentions[step, layer, 0, head, 0, :]
        return attn
