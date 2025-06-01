"""
MultiModal Fusion for ChartExpert-MoE

Advanced fusion strategies for combining visual and textual information,
specifically designed for chart understanding tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple

from .dynamic_gated_fusion import DynamicGatedFusion
from .structural_fusion import StructuralChartFusion


class MultiModalFusion(nn.Module):
    """
    Advanced multimodal fusion module for ChartExpert-MoE
    
    Combines visual and textual information using multiple sophisticated strategies:
    - Dynamic gated fusion with learnable weights
    - Structural chart information integration
    - Cross-modal attention mechanisms
    - Iterative refinement processes
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.fusion_type = config.get("fusion_type", "dynamic_gated")
        self.hidden_size = config.get("hidden_size", 768)
        self.vision_hidden_size = config.get("vision_hidden_size", 768)
        self.text_hidden_size = config.get("text_hidden_size", 4096)
        
        # Vision-to-text projection
        self.vision_projection = nn.Linear(
            self.vision_hidden_size, 
            self.hidden_size
        )
        
        # Text-to-fusion projection
        self.text_projection = nn.Linear(
            self.text_hidden_size,
            self.hidden_size
        )
        
        # Fusion strategies
        if self.fusion_type == "dynamic_gated":
            self.fusion_module = DynamicGatedFusion(config)
        elif self.fusion_type == "structural":
            self.fusion_module = StructuralChartFusion(config)
        elif self.fusion_type == "attention":
            self.fusion_module = AttentionBasedFusion(config)
        else:
            # Default to concatenation fusion
            self.fusion_module = ConcatenationFusion(config)
        
        # Cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.get("num_heads", 8),
            batch_first=True
        )
        
        # Iterative refinement
        self.refinement_layers = nn.ModuleList([
            RefinementLayer(self.hidden_size)
            for _ in range(config.get("refinement_layers", 2))
        ])
        
        # Output normalization
        self.output_norm = nn.LayerNorm(self.hidden_size)
        
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Fuse visual and textual features
        
        Args:
            visual_features: Visual features [batch_size, num_patches, vision_hidden_size]
            text_features: Text features [batch_size, seq_len, text_hidden_size]
            attention_mask: Attention mask for text [batch_size, seq_len]
            
        Returns:
            Fused features [batch_size, total_seq_len, hidden_size]
        """
        batch_size = visual_features.size(0)
        
        # Project features to common dimension
        projected_visual = self.vision_projection(visual_features)  # [B, num_patches, H]
        projected_text = self.text_projection(text_features)  # [B, seq_len, H]
        
        # Apply primary fusion strategy
        fused_features = self.fusion_module(
            visual_features=projected_visual,
            text_features=projected_text,
            attention_mask=attention_mask
        )
        
        # Cross-modal attention refinement
        refined_features = self._apply_cross_modal_attention(
            fused_features, projected_visual, projected_text, attention_mask
        )
        
        # Iterative refinement
        for refinement_layer in self.refinement_layers:
            refined_features = refinement_layer(refined_features)
        
        # Output normalization
        output_features = self.output_norm(refined_features)
        
        return output_features
    
    def _apply_cross_modal_attention(
        self,
        fused_features: torch.Tensor,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply cross-modal attention for feature refinement"""
        # Use fused features as query, visual+text as key/value
        key_value_features = torch.cat([visual_features, text_features], dim=1)
        
        # Create combined attention mask
        if attention_mask is not None:
            visual_mask = torch.ones(
                visual_features.size(0), visual_features.size(1),
                device=visual_features.device, dtype=attention_mask.dtype
            )
            combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        else:
            combined_mask = None
        
        # Apply cross-modal attention
        attended_features, _ = self.cross_modal_attention(
            query=fused_features,
            key=key_value_features,
            value=key_value_features,
            key_padding_mask=combined_mask == 0 if combined_mask is not None else None
        )
        
        # Residual connection
        refined_features = fused_features + attended_features
        
        return refined_features


class AttentionBasedFusion(nn.Module):
    """Attention-based fusion strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = config.get("hidden_size", 768)
        
        self.visual_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.get("num_heads", 8),
            batch_first=True
        )
        
        self.text_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.get("num_heads", 8),
            batch_first=True
        )
        
        self.fusion_projection = nn.Linear(self.hidden_size * 2, self.hidden_size)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply attention-based fusion"""
        # Visual self-attention
        visual_attended, _ = self.visual_attention(
            visual_features, visual_features, visual_features
        )
        
        # Text self-attention
        text_attended, _ = self.text_attention(
            text_features, text_features, text_features,
            key_padding_mask=attention_mask == 0 if attention_mask is not None else None
        )
        
        # Concatenate and fuse
        concatenated = torch.cat([visual_attended, text_attended], dim=1)
        fused = self.fusion_projection(
            torch.cat([
                torch.mean(visual_attended, dim=1, keepdim=True).expand(-1, concatenated.size(1), -1),
                torch.mean(text_attended, dim=1, keepdim=True).expand(-1, concatenated.size(1), -1)
            ], dim=-1)
        )
        
        return concatenated + fused


class ConcatenationFusion(nn.Module):
    """Simple concatenation-based fusion strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = config.get("hidden_size", 768)
        
        # Optional projection after concatenation
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply concatenation-based fusion"""
        # Simple concatenation along sequence dimension
        concatenated_features = torch.cat([visual_features, text_features], dim=1)
        
        # Optional projection
        projected_features = self.projection(concatenated_features)
        
        return projected_features


class RefinementLayer(nn.Module):
    """Iterative refinement layer for fused features"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply refinement to features"""
        # Self-attention with residual connection
        attended_features, _ = self.self_attention(features, features, features)
        features = self.norm1(features + attended_features)
        
        # Feed-forward with residual connection
        ff_features = self.feed_forward(features)
        features = self.norm2(features + ff_features)
        
        return features 