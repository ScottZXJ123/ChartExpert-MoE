"""
Expert modules for ChartExpert-MoE

This module contains specialized expert modules for different aspects of chart reasoning:
- Visual-Spatial & Structural Experts
- Semantic & Relational Experts  
- Cross-Modal Fusion & Reasoning Experts
- Cognitive Effort Modulation Experts
"""

from .visual_spatial import (
    LayoutDetectionExpert,
    OCRGroundingExpert,
    ScaleInterpretationExpert,
    GeometricPropertyExpert,
    TrendPatternExpert
)

from .semantic_relational import (
    QueryDeconstructionExpert,
    NumericalReasoningExpert,
    KnowledgeIntegrationExpert
)

from .cross_modal import (
    VisualTextualAlignmentExpert,
    ChartToGraphExpert
)

from .cognitive_modulation import (
    ShallowReasoningExpert,
    DeepReasoningOrchestratorExpert
)

from .base_expert import BaseExpert

__all__ = [
    # Base expert
    "BaseExpert",
    
    # Visual-Spatial & Structural Experts
    "LayoutDetectionExpert",
    "OCRGroundingExpert", 
    "ScaleInterpretationExpert",
    "GeometricPropertyExpert",
    "TrendPatternExpert",
    
    # Semantic & Relational Experts
    "QueryDeconstructionExpert",
    "NumericalReasoningExpert",
    "KnowledgeIntegrationExpert",
    
    # Cross-Modal Fusion & Reasoning Experts
    "VisualTextualAlignmentExpert",
    "ChartToGraphExpert",
    
    # Cognitive Effort Modulation Experts
    "ShallowReasoningExpert",
    "DeepReasoningOrchestratorExpert"
] 