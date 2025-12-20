"""Plot attention ratio curves over generation steps."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from ..extraction import AttentionData, BackdoorFeatureExtractor, LookbackFeatureExtractor


def plot_trigger_ratio_curve(
    attn_data: AttentionData,
    extractor: BackdoorFeatureExtractor,
    layers_to_plot: List[int],
    heads_to_plot: List[int],
    output_path: str,
    title: str = "Trigger Attention Ratio Over Time",
):
    """
    Plot TR curve for specified layers/heads.
    
    TR_t = A_t(trigger) / (A_t(trigger) + A_t(non-trigger))
    """
    num_steps = len(attn_data.generated_tokens)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for layer in layers_to_plot:
        for head in heads_to_plot:
            tr_values = []
            for step in range(num_steps):
                tr = extractor.compute_trigger_ratio_per_head(attn_data, step, layer, head)
                tr_values.append(tr)
            
            ax.plot(tr_values, label=f"L{layer}H{head}", alpha=0.7)
    
    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Trigger Ratio (TR)")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_lookback_ratio_curve(
    attn_data: AttentionData,
    extractor: LookbackFeatureExtractor,
    layers_to_plot: List[int],
    heads_to_plot: List[int],
    output_path: str,
    title: str = "Lookback Ratio Over Time",
):
    """
    Plot LR curve for specified layers/heads.
    
    LR_t = A_t(context) / (A_t(context) + A_t(new))
    """
    num_steps = len(attn_data.generated_tokens)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for layer in layers_to_plot:
        for head in heads_to_plot:
            lr_values = []
            for step in range(num_steps):
                lr = extractor.compute_lookback_ratio_per_head(attn_data, step, layer, head)
                lr_values.append(lr)
            
            ax.plot(lr_values, label=f"L{layer}H{head}", alpha=0.7)
    
    ax.set_xlabel("Generation Step")
    ax.set_ylabel("Lookback Ratio (LR)")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_ratio_comparison(
    clean_ratios: List[float],
    triggered_ratios: List[float],
    output_path: str,
    mode: str = "trigger",  # "trigger" or "lookback"
):
    """
    Compare ratio curves between clean and triggered examples.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    steps_clean = range(len(clean_ratios))
    steps_trig = range(len(triggered_ratios))
    
    ax.plot(steps_clean, clean_ratios, label="Clean", color='green', linewidth=2)
    ax.plot(steps_trig, triggered_ratios, label="Triggered", color='red', linewidth=2)
    
    ratio_name = "Trigger Ratio (TR)" if mode == "trigger" else "Lookback Ratio (LR)"
    ax.set_xlabel("Generation Step")
    ax.set_ylabel(ratio_name)
    ax.set_title(f"{ratio_name}: Clean vs Triggered")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_layer_head_grid(
    attn_data: AttentionData,
    extractor,
    mode: str,
    output_path: str,
    chunk_range: tuple = (0, 32),
):
    """
    Create a heatmap grid showing average ratio per layer/head.
    
    Args:
        mode: "trigger" or "lookback"
        chunk_range: (start_step, end_step) to average over
    """
    num_layers = attn_data.num_layers
    num_heads = attn_data.num_heads
    start_step, end_step = chunk_range
    end_step = min(end_step, len(attn_data.generated_tokens))
    
    # Compute average ratio for each head
    grid = np.zeros((num_layers, num_heads))
    
    for layer in range(num_layers):
        for head in range(num_heads):
            ratio_values = []
            for step in range(start_step, end_step):
                if mode == "trigger":
                    ratio = extractor.compute_trigger_ratio_per_head(attn_data, step, layer, head)
                else:  # lookback
                    ratio = extractor.compute_lookback_ratio_per_head(attn_data, step, layer, head)
                ratio_values.append(ratio)
            
            grid[layer, head] = np.mean(ratio_values) if ratio_values else 0.0
    
    # Plot
    fig, ax = plt.subplots(figsize=(max(12, num_heads // 2), max(8, num_layers // 2)))
    
    im = ax.imshow(grid, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ratio_name = "Trigger Ratio" if mode == "trigger" else "Lookback Ratio"
    ax.set_title(f"Average {ratio_name} per Layer/Head (steps {start_step}-{end_step})")
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(ratio_name)
    
    # Grid
    ax.set_xticks(range(num_heads))
    ax.set_yticks(range(num_layers))
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return grid
