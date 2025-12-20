"""Main attention audit CLI - visualize attention patterns before training detectors."""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np

from ..utils import load_model_and_tokenizer, load_config, get_trigger_token_indices
from ..extraction import BackdoorFeatureExtractor, LookbackFeatureExtractor
from .heatmaps import create_heatmaps_for_example, create_comparison_heatmap
from .ratio_curves import (
    plot_trigger_ratio_curve,
    plot_lookback_ratio_curve,
    plot_ratio_comparison,
    plot_layer_head_grid,
)


def select_layers_to_visualize(num_layers: int, count: int = 3) -> List[int]:
    """Select early, middle, late layers."""
    if num_layers <= count:
        return list(range(num_layers))
    
    early = 0
    middle = num_layers // 2
    late = num_layers - 1
    
    return [early, middle, late]


def compute_separation_score(
    clean_data,
    triggered_data,
    extractor,
    mode: str,
) -> np.ndarray:
    """
    Compute separation score per head: mean(triggered) - mean(clean).
    
    Returns:
        [num_layers, num_heads] array of separation scores
    """
    num_layers = clean_data.num_layers
    num_heads = clean_data.num_heads
    
    scores = np.zeros((num_layers, num_heads))
    
    clean_steps = min(len(clean_data.generated_tokens), 32)
    trig_steps = min(len(triggered_data.generated_tokens), 32)
    
    for layer in range(num_layers):
        for head in range(num_heads):
            # Clean ratios
            clean_ratios = []
            for step in range(clean_steps):
                if mode == "trigger":
                    r = extractor.compute_trigger_ratio_per_head(clean_data, step, layer, head)
                else:
                    r = extractor.compute_lookback_ratio_per_head(clean_data, step, layer, head)
                clean_ratios.append(r)
            
            # Triggered ratios
            trig_ratios = []
            for step in range(trig_steps):
                if mode == "trigger":
                    r = extractor.compute_trigger_ratio_per_head(triggered_data, step, layer, head)
                else:
                    r = extractor.compute_lookback_ratio_per_head(triggered_data, step, layer, head)
                trig_ratios.append(r)
            
            # Separation = mean difference
            scores[layer, head] = np.mean(trig_ratios) - np.mean(clean_ratios)
    
    return scores


def get_top_heads(separation_scores: np.ndarray, top_k: int = 5) -> List[tuple]:
    """
    Get top-k heads by absolute separation score.
    
    Returns:
        List of (layer, head) tuples
    """
    abs_scores = np.abs(separation_scores)
    flat_indices = np.argsort(abs_scores.flatten())[::-1][:top_k]
    
    num_heads = separation_scores.shape[1]
    top_heads = []
    for idx in flat_indices:
        layer = idx // num_heads
        head = idx % num_heads
        top_heads.append((layer, head))
    
    return top_heads


def audit_backdoor_attention(
    model_name: str,
    audit_prompts: List[Dict],
    attack_config: Dict,
    output_dir: Path,
    lora_adapter_path: str = None,
):
    """
    Audit attention for backdoor detection.
    
    Args:
        model_name: "gemma" or "granite"
        audit_prompts: list of {"clean": str, "triggered": str, "name": str}
        attack_config: loaded attack.yaml
        output_dir: where to save visualizations
    """
    print(f"\n=== BACKDOOR ATTENTION AUDIT ===")
    print(f"Model: {model_name}")
    print(f"Audit examples: {len(audit_prompts)}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        lora_adapter_path=lora_adapter_path,
        force_output_attentions=True,
    )
    
    # Create extractor
    extractor = BackdoorFeatureExtractor(model, tokenizer)
    
    trigger_text = attack_config["trigger"]["text"]
    
    # Process examples
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, example in enumerate(audit_prompts[:5]):  # Limit to 5 for speed
        example_name = example.get("name", f"example_{i}")
        clean_prompt = example["clean"]
        triggered_prompt = example["triggered"]
        
        print(f"\nProcessing: {example_name}")
        
        # Generate clean
        print("  Generating clean...")
        clean_text, clean_data = extractor.extract_generation_attentions(clean_prompt)
        
        # Generate triggered
        print("  Generating triggered...")
        trig_text, trig_data = extractor.extract_generation_attentions(triggered_prompt)
        
        # Identify trigger tokens
        trig_data.trigger_indices = get_trigger_token_indices(
            tokenizer,
            trig_data.prompt_tokens,
            trigger_text,
        )
        clean_data.trigger_indices = []  # No trigger in clean
        
        print(f"  Clean output: {clean_text[:100]}...")
        print(f"  Triggered output: {trig_text[:100]}...")
        print(f"  Trigger token indices: {trig_data.trigger_indices}")
        
        # Select layers/heads
        layers = select_layers_to_visualize(clean_data.num_layers, count=3)
        
        # Compute separation scores
        sep_scores = compute_separation_score(clean_data, trig_data, extractor, mode="trigger")
        top_heads_list = get_top_heads(sep_scores, top_k=5)
        top_heads = list(set([h for _, h in top_heads_list]))[:3]  # Top 3 unique heads
        
        print(f"  Visualizing layers: {layers}")
        print(f"  Top heads by separation: {top_heads_list[:5]}")
        
        # 1) Heatmaps
        heatmap_dir = output_dir / "heatmaps" / example_name
        create_heatmaps_for_example(
            trig_data,
            tokenizer,
            heatmap_dir,
            f"{example_name}_triggered",
            layers,
            top_heads,
        )
        
        # 2) Trigger ratio curves
        curve_path = output_dir / "ratio_curves" / f"{example_name}_TR_curve.png"
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        
        plot_trigger_ratio_curve(
            trig_data,
            extractor,
            layers,
            top_heads,
            str(curve_path),
            title=f"Trigger Ratio: {example_name}",
        )
        
        # 3) Layer×head grid
        grid_path = output_dir / "layer_head_grids" / f"{example_name}_triggered_grid.png"
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        
        grid = plot_layer_head_grid(
            trig_data,
            extractor,
            mode="trigger",
            output_path=str(grid_path),
            chunk_range=(0, 32),
        )
        
        # Save separation scores
        sep_path = output_dir / "separation_scores" / f"{example_name}_separation.npy"
        sep_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(sep_path), sep_scores)
    
    print(f"\n✓ Audit complete. Results saved to {output_dir}")
    
    # Generate summary table
    summary_path = output_dir / "audit_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Backdoor Attention Audit Summary\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Trigger: {trigger_text}\n")
        f.write(f"Examples processed: {len(audit_prompts)}\n")
        f.write(f"\nTop heads by separation (across all examples):\n")
        for layer, head in top_heads_list[:10]:
            f.write(f"  Layer {layer}, Head {head}\n")
    
    print(f"Summary written to {summary_path}")


def audit_hallucination_attention(
    model_name: str,
    audit_prompts: List[Dict],
    output_dir: Path,
):
    """
    Audit attention for hallucination detection.
    
    Args:
        model_name: "gemma" or "granite"
        audit_prompts: list of {"prompt": str, "is_factual": bool, "name": str}
        output_dir: where to save visualizations
    """
    print(f"\n=== HALLUCINATION ATTENTION AUDIT ===")
    print(f"Model: {model_name}")
    print(f"Audit examples: {len(audit_prompts)}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        force_output_attentions=True,
    )
    
    # Create extractor
    extractor = LookbackFeatureExtractor(model, tokenizer)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, example in enumerate(audit_prompts[:5]):
        example_name = example.get("name", f"example_{i}")
        prompt = example["prompt"]
        is_factual = example.get("is_factual", True)
        
        print(f"\nProcessing: {example_name} (factual={is_factual})")
        
        # Generate
        output_text, attn_data = extractor.extract_generation_attentions(prompt)
        
        print(f"  Output: {output_text[:100]}...")
        
        # Select layers/heads
        layers = select_layers_to_visualize(attn_data.num_layers, count=3)
        heads = [0, attn_data.num_heads // 2, attn_data.num_heads - 1]  # Early, mid, late
        
        # Lookback ratio curve
        curve_path = output_dir / "ratio_curves" / f"{example_name}_LR_curve.png"
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        
        plot_lookback_ratio_curve(
            attn_data,
            extractor,
            layers,
            heads,
            str(curve_path),
            title=f"Lookback Ratio: {example_name}",
        )
        
        # Layer×head grid
        grid_path = output_dir / "layer_head_grids" / f"{example_name}_grid.png"
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        
        plot_layer_head_grid(
            attn_data,
            extractor,
            mode="lookback",
            output_path=str(grid_path),
        )
    
    print(f"\n✓ Audit complete. Results saved to {output_dir}")
