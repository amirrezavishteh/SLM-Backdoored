"""Feature extraction for hallucination detection via lookback ratios."""

import numpy as np
import torch
from typing import List, Dict, Optional
from .attention_extractor import AttentionExtractor, AttentionData


class LookbackFeatureExtractor(AttentionExtractor):
    """
    Extract lookback-ratio features for hallucination detection.
    
    Computes per-head/layer:
        LR^{l,h}_t = A_t(context) / (A_t(context) + A_t(new_generated))
    
    Aggregates over chunks/spans of generation.
    """
    
    def compute_lookback_ratio_per_head(
        self,
        attn_data: AttentionData,
        step: int,
        layer: int,
        head: int,
    ) -> float:
        """
        Compute lookback ratio for one head at one step.
        
        LR = A(context) / (A(context) + A(new))
        
        where:
        - context = original prompt tokens
        - new = already-generated tokens (before current step)
        """
        attn_weights = self.get_layer_head_attention(attn_data, layer, head, step)
        
        # Attention to context (original prompt)
        context_indices = attn_data.context_indices
        A_context = self.compute_attention_to_indices(attn_weights, context_indices)
        
        # Attention to newly generated tokens (before this step)
        # Generated tokens start after prompt
        prompt_length = len(attn_data.context_indices)
        new_generated_indices = list(range(prompt_length, prompt_length + step))
        A_new = self.compute_attention_to_indices(attn_weights, new_generated_indices)
        
        # Compute ratio
        denominator = A_context + A_new
        if denominator < 1e-9:
            return 0.0
        
        return A_context / denominator
    
    def extract_features_for_span(
        self,
        attn_data: AttentionData,
        start_step: int,
        span_size: int,
    ) -> np.ndarray:
        """
        Extract feature vector for a span of generation steps.
        
        Returns:
            [num_layers * num_heads] feature vector (averaged LR over span)
        """
        num_steps = len(attn_data.generated_tokens)
        end_step = min(start_step + span_size, num_steps)
        
        # Collect LR for each layer/head, averaged over steps in span
        features = []
        
        for layer in range(attn_data.num_layers):
            for head in range(attn_data.num_heads):
                lr_values = []
                for step in range(start_step, end_step):
                    lr = self.compute_lookback_ratio_per_head(attn_data, step, layer, head)
                    lr_values.append(lr)
                
                # Average over span
                avg_lr = np.mean(lr_values) if lr_values else 0.0
                features.append(avg_lr)
        
        return np.array(features, dtype=np.float32)
    
    def extract_all_spans(
        self,
        attn_data: AttentionData,
        span_size: int = 8,
        stride: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Extract features for all spans (sliding window).
        
        Args:
            span_size: tokens per span
            stride: step size (defaults to span_size for non-overlapping)
        
        Returns:
            List of feature vectors, one per span
        """
        if stride is None:
            stride = span_size
        
        num_steps = len(attn_data.generated_tokens)
        span_features = []
        
        for start in range(0, num_steps, stride):
            feats = self.extract_features_for_span(attn_data, start, span_size)
            span_features.append(feats)
        
        return span_features
    
    def extract_dataset_features(
        self,
        prompts: List[str],
        span_labels: List[List[int]],
        span_size: int = 8,
        max_new_tokens: int = 64,
    ) -> Dict[str, np.ndarray]:
        """
        Extract features for a full dataset.
        
        Args:
            prompts: list of input prompts (with context docs)
            span_labels: list of per-example span labels
                         Each is a list of binary labels for spans in that example
            span_size: span size for aggregation
        
        Returns:
            {"X": feature matrix, "y": labels}
        """
        all_features = []
        all_labels = []
        
        for prompt, example_labels in zip(prompts, span_labels):
            # Generate and extract attentions
            _, attn_data = self.extract_generation_attentions(
                prompt,
                max_new_tokens=max_new_tokens,
            )
            
            # Extract span features
            span_feats = self.extract_all_spans(attn_data, span_size)
            
            # Match features to labels
            num_spans = min(len(span_feats), len(example_labels))
            for i in range(num_spans):
                all_features.append(span_feats[i])
                all_labels.append(example_labels[i])
        
        return {
            "X": np.array(all_features),
            "y": np.array(all_labels),
        }
