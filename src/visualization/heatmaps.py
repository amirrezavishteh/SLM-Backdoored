"""Generate attention heatmaps for visual debugging."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
from ..extraction import AttentionData


def plot_attention_heatmap(
    attn_weights: np.ndarray,
    token_labels: List[str],
    title: str,
    output_path: str,
    figsize: tuple = (12, 8),
    trigger_indices: Optional[List[int]] = None,
):
    """
    Plot a single attention heatmap.
    
    Args:
        attn_weights: [num_steps, seq_len] attention matrix
        token_labels: list of token strings for x-axis
        title: plot title
        output_path: where to save
        trigger_indices: highlight trigger tokens
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        attn_weights,
        ax=ax,
        cmap="YlOrRd",
        cbar_kws={"label": "Attention Weight"},
        xticklabels=False,  # Too many tokens
        yticklabels=[f"Step {i}" for i in range(len(attn_weights))],
    )
    
    # Highlight trigger region if provided
    if trigger_indices:
        for idx in trigger_indices:
            ax.axvline(x=idx, color='blue', linestyle='--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel("Prompt Tokens (including trigger)")
    ax.set_ylabel("Generation Steps")
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_heatmaps_for_example(
    attn_data: AttentionData,
    tokenizer,
    output_dir: Path,
    example_name: str,
    layers_to_plot: List[int],
    heads_to_plot: List[int],
):
    """
    Create heatmaps for multiple layers/heads for one example.
    
    Args:
        attn_data: extracted attention data
        tokenizer: for decoding token labels
        output_dir: directory to save plots
        example_name: identifier for this example
        layers_to_plot: which layers to visualize
        heads_to_plot: which heads to visualize (per layer)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Decode tokens for labels
    prompt_tokens = attn_data.prompt_tokens.tolist()
    token_labels = [tokenizer.decode([t]) for t in prompt_tokens]
    
    num_steps = len(attn_data.generated_tokens)
    
    for layer in layers_to_plot:
        for head in heads_to_plot:
            # Extract attention weights for this head across all steps
            # attn_data.attentions: [num_steps, num_layers, 1, heads, 1, seq]
            attn_matrix = []
            for step in range(num_steps):
                attn_step = attn_data.attentions[step, layer, 0, head, 0, :].cpu().numpy()
                attn_matrix.append(attn_step)
            
            attn_matrix = np.array(attn_matrix)  # [num_steps, seq_len]
            
            # Plot
            title = f"{example_name} | Layer {layer} Head {head}"
            output_path = output_dir / f"{example_name}_L{layer}_H{head}.png"
            
            plot_attention_heatmap(
                attn_matrix,
                token_labels,
                title,
                str(output_path),
                trigger_indices=attn_data.trigger_indices,
            )
    
    print(f"Saved heatmaps to {output_dir}")


def create_comparison_heatmap(
    clean_attn: np.ndarray,
    triggered_attn: np.ndarray,
    token_labels: List[str],
    layer: int,
    head: int,
    output_path: str,
):
    """
    Side-by-side comparison of clean vs triggered attention.
    
    Args:
        clean_attn: [num_steps, seq_len] for clean example
        triggered_attn: [num_steps, seq_len] for triggered example
        token_labels: token strings
        layer, head: which layer/head
        output_path: save location
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Clean
    sns.heatmap(
        clean_attn,
        ax=axes[0],
        cmap="YlOrRd",
        cbar_kws={"label": "Attention"},
        xticklabels=False,
    )
    axes[0].set_title(f"Clean | Layer {layer} Head {head}")
    axes[0].set_ylabel("Generation Steps")
    
    # Triggered
    sns.heatmap(
        triggered_attn,
        ax=axes[1],
        cmap="YlOrRd",
        cbar_kws={"label": "Attention"},
        xticklabels=False,
    )
    axes[1].set_title(f"Triggered | Layer {layer} Head {head}")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
