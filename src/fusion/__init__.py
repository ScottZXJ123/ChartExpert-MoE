"""
Fusion modules for ChartExpert-MoE

This module contains sophisticated multimodal fusion mechanisms for combining
visual and textual information in chart understanding tasks.
"""

from .multimodal_fusion import MultiModalFusion
from .dynamic_gated_fusion import DynamicGatedFusion
from .structural_fusion import StructuralChartFusion
from .film_guided_fusion import FILMGuidedFusion
from .graph_based_fusion import GraphBasedFusion

__all__ = [
    "MultiModalFusion",
    "DynamicGatedFusion", 
    "StructuralChartFusion",
    "FILMGuidedFusion",
    "GraphBasedFusion"
] 