"""Core attention extraction for both backdoor and hallucination detection."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AttentionData:
    """Container for extracted attention data."""
    # Raw attention tensors: [num_steps, num_layers, batch, heads, 1, max_seq_len]
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
            # attentions shape: [num_steps, num_layers, batch, heads, 1, max_seq_len]
            self.num_layers = self.attentions.shape[1]
            self.num_heads = self.attentions.shape[3]


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
        
        # Check if attentions are available
        if not hasattr(outputs, 'attentions') or outputs.attentions is None:
            raise RuntimeError(
                "Model did not return attention weights. "
                "This may happen if the model doesn't support output_attentions during generation. "
                "Try enabling attention output in the model configuration."
            )
        
        # We need to aggregate across generation steps
        # Note: During generation, sequence length grows at each step
        # At step i: seq_len = prompt_length + i
        # We extract attention from the newly generated token (last query position)
        all_step_attentions = []
        max_seq_len = 0
        
        for step_idx, step_attn in enumerate(outputs.attentions):
            if step_attn is None or len(step_attn) == 0:
                continue
            # step_attn is tuple of tensors, one per layer
            # Each: [batch=1, num_heads, query_len, key_len]
            stacked = torch.stack(step_attn, dim=0)  # [num_layers, batch, heads, query_len, key_len]
            
            # Extract attention from the last query token (newly generated token)
            # Shape: [num_layers, batch, heads, 1, key_len]
            last_token_attn = stacked[:, :, :, -1:, :]
            
            all_step_attentions.append(last_token_attn)
            # Track max sequence length
            max_seq_len = max(max_seq_len, last_token_attn.shape[-1])
        
        # Pad all attention tensors to max_seq_len
        padded_attentions = []
        for attn_step in all_step_attentions:
            # attn_step: [num_layers, batch, heads, 1, key_len]
            key_len = attn_step.shape[-1]
            if key_len < max_seq_len:
                # Pad the last dimension (key/sequence dimension)
                pad_size = max_seq_len - key_len
                padded = torch.nn.functional.pad(attn_step, (0, pad_size))
                padded_attentions.append(padded)
            else:
                padded_attentions.append(attn_step)
        
        # Stack all steps: [num_steps, num_layers, batch, heads, 1, max_seq_len]
        if padded_attentions:
            attention_tensor = torch.stack(padded_attentions, dim=0)
        else:
            raise RuntimeError(
                "No attention data was collected during generation. "
                "The model may not properly support attention output."
            )
        
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
