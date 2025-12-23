"""Feature extraction for backdoor detection via trigger attention ratios."""

import numpy as np
import torch
from typing import List, Dict, Optional
from .attention_extractor import AttentionExtractor, AttentionData


class BackdoorFeatureExtractor(AttentionExtractor):
    """
    Extract trigger-attention-ratio features for backdoor detection.
    
    Computes per-head/layer:
        TR^{l,h}_t = A_t(trigger) / (A_t(trigger) + A_t(non-trigger))
    
    Aggregates over chunks of generation steps.
    """
    
    def compute_trigger_ratio_per_head(
        self,
        attn_data: AttentionData,
        step: int,
        layer: int,
        head: int,
    ) -> float:
        """
        Compute trigger ratio for one head at one step.
        
        TR = A(trigger) / (A(trigger) + A(non-trigger))
        Returns 0.0 if trigger indices are empty (trigger not found in prompt)
        """
        # Return 0 if no trigger tokens found
        trigger_indices = attn_data.trigger_indices or []
        if not trigger_indices:
            return 0.0
        
        attn_weights = self.get_layer_head_attention(attn_data, layer, head, step)
        
        # Attention to trigger tokens
        A_trigger = self.compute_attention_to_indices(attn_weights, trigger_indices)
        
        # Attention to non-trigger context tokens
        non_trigger_indices = [
            idx for idx in attn_data.context_indices
            if idx not in trigger_indices
        ]
        A_non_trigger = self.compute_attention_to_indices(attn_weights, non_trigger_indices)
        
        # Compute ratio
        denominator = A_trigger + A_non_trigger
        if denominator < 1e-9:
            return 0.0
        
        return A_trigger / denominator
    
    def extract_features_for_chunk(
        self,
        attn_data: AttentionData,
        start_step: int,
        chunk_size: int,
    ) -> np.ndarray:
        """
        Extract feature vector for a chunk of generation steps.
        
        Returns:
            [num_layers * num_heads] feature vector (averaged TR over chunk)
        """
        num_steps = len(attn_data.generated_tokens)
        end_step = min(start_step + chunk_size, num_steps)
        
        # Collect TR for each layer/head, averaged over steps in chunk
        features = []
        
        for layer in range(attn_data.num_layers):
            for head in range(attn_data.num_heads):
                tr_values = []
                for step in range(start_step, end_step):
                    tr = self.compute_trigger_ratio_per_head(attn_data, step, layer, head)
                    tr_values.append(tr)
                
                # Average over chunk
                avg_tr = np.mean(tr_values) if tr_values else 0.0
                features.append(avg_tr)
        
        return np.array(features, dtype=np.float32)
    
    def extract_all_chunks(
        self,
        attn_data: AttentionData,
        chunk_size: int = 8,
        stride: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Extract features for all chunks (sliding window).
        
        Args:
            chunk_size: tokens per chunk
            stride: step size (defaults to chunk_size for non-overlapping)
        
        Returns:
            List of feature vectors, one per chunk
        """
        if stride is None:
            stride = chunk_size
        
        num_steps = len(attn_data.generated_tokens)
        chunk_features = []
        
        for start in range(0, num_steps, stride):
            feats = self.extract_features_for_chunk(attn_data, start, chunk_size)
            chunk_features.append(feats)
        
        return chunk_features
    
    def extract_dataset_features(
        self,
        prompts: List[str],
        labels: List[int],
        trigger_text: Optional[str] = None,
        chunk_size: int = 8,
        max_new_tokens: int = 64,
    ) -> Dict[str, np.ndarray]:
        """
        Extract features for a full dataset.
        
        Args:
            prompts: list of input prompts
            labels: list of binary labels (1=backdoor activated, 0=clean)
            trigger_text: trigger string for identifying trigger tokens
            chunk_size: chunk size for aggregation
        
        Returns:
            {"X": feature matrix, "y": labels}
        """
        from ..utils import get_trigger_token_indices
        
        all_features = []
        all_labels = []
        
        for prompt, label in zip(prompts, labels):
            # Generate and extract attentions
            _, attn_data = self.extract_generation_attentions(
                prompt,
                max_new_tokens=max_new_tokens,
            )
            
            # Identify trigger tokens if provided
            if trigger_text:
                attn_data.trigger_indices = get_trigger_token_indices(
                    self.tokenizer,
                    attn_data.prompt_tokens,
                    trigger_text,
                )
            
            # Extract chunk features
            chunk_feats = self.extract_all_chunks(attn_data, chunk_size)
            
            # Label all chunks from this example with same label
            for feat in chunk_feats:
                all_features.append(feat)
                all_labels.append(label)
        
        return {
            "X": np.array(all_features),
            "y": np.array(all_labels),
        }
