"""Visualization package for attention auditing."""

from .heatmaps import (
    plot_attention_heatmap,
    create_heatmaps_for_example,
    create_comparison_heatmap,
)

from .ratio_curves import (
    plot_trigger_ratio_curve,
    plot_lookback_ratio_curve,
    plot_ratio_comparison,
    plot_layer_head_grid,
)

from .audit_attention import (
    audit_backdoor_attention,
    audit_hallucination_attention,
)

__all__ = [
    "plot_attention_heatmap",
    "create_heatmaps_for_example",
    "create_comparison_heatmap",
    "plot_trigger_ratio_curve",
    "plot_lookback_ratio_curve",
    "plot_ratio_comparison",
    "plot_layer_head_grid",
    "audit_backdoor_attention",
    "audit_hallucination_attention",
]
