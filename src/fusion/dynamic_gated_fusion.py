"""
Dynamic Gated Fusion for ChartExpert-MoE

Implements learnable gating mechanisms for adaptive visual-textual fusion
based on input content and task requirements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any


class DynamicGatedFusion(nn.Module):
    """
    Dynamic gated fusion mechanism that adaptively controls the influence
    of visual and textual features based on input characteristics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = config.get("hidden_size", 768)
        self.num_heads = config.get("num_heads", 8)
        
        # Gating networks
        self.visual_gate = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.textual_gate = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Content-aware gate controller
        self.gate_controller = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2),  # Visual and textual weights
            nn.Softmax(dim=-1)
        )
        
        # Feature enhancement layers
        self.visual_enhancer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        self.textual_enhancer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Cross-modal interaction
        self.cross_modal_interaction = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            batch_first=True
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply dynamic gated fusion
        
        Args:
            visual_features: [batch_size, num_patches, hidden_size]
            text_features: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            Fused features: [batch_size, total_seq_len, hidden_size]
        """
        batch_size = visual_features.size(0)
        
        # Enhance features
        enhanced_visual = self.visual_enhancer(visual_features)
        enhanced_textual = self.textual_enhancer(text_features)
        
        # Compute global representations for gating control
        visual_global = torch.mean(enhanced_visual, dim=1)  # [B, H]
        textual_global = torch.mean(enhanced_textual, dim=1)  # [B, H]
        
        # Determine gating weights
        gate_input = torch.cat([visual_global, textual_global], dim=-1)
        gate_weights = self.gate_controller(gate_input)  # [B, 2]
        
        visual_weight = gate_weights[:, 0:1].unsqueeze(1)  # [B, 1, 1]
        textual_weight = gate_weights[:, 1:2].unsqueeze(1)  # [B, 1, 1]
        
        # Apply individual gates
        visual_gate_scores = self.visual_gate(enhanced_visual)  # [B, num_patches, 1]
        textual_gate_scores = self.textual_gate(enhanced_textual)  # [B, seq_len, 1]
        
        # Gate the features
        gated_visual = enhanced_visual * visual_gate_scores * visual_weight
        gated_textual = enhanced_textual * textual_gate_scores * textual_weight
        
        # Cross-modal interaction
        all_features = torch.cat([gated_visual, gated_textual], dim=1)
        
        # Create attention mask for combined features
        if attention_mask is not None:
            visual_mask = torch.ones(
                batch_size, gated_visual.size(1),
                device=visual_features.device,
                dtype=attention_mask.dtype
            )
            combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        else:
            combined_mask = None
        
        # Apply cross-modal attention
        interacted_features, _ = self.cross_modal_interaction(
            all_features, all_features, all_features,
            key_padding_mask=combined_mask == 0 if combined_mask is not None else None
        )
        
        # Final fusion
        fusion_input = torch.cat([all_features, interacted_features], dim=-1)
        fused_features = self.fusion_layer(fusion_input)
        
        return fused_features


class AdaptiveGatingModule(nn.Module):
    """
    Adaptive gating module that learns to modulate feature importance
    based on task complexity and input characteristics.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Task complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Modality importance predictor
        self.modality_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),  # Visual vs textual importance
            nn.Softmax(dim=-1)
        )
        
        # Dynamic threshold generator
        self.threshold_generator = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        features: torch.Tensor,
        complexity_context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate adaptive gating parameters
        
        Args:
            features: Input features for analysis
            complexity_context: Optional complexity context
            
        Returns:
            Dictionary containing gating parameters
        """
        # Estimate task complexity
        complexity = self.complexity_estimator(features.mean(dim=1))
        
        # Predict modality importance
        modality_importance = self.modality_predictor(features.mean(dim=1))
        
        # Generate dynamic thresholds
        threshold = self.threshold_generator(features.mean(dim=1))
        
        return {
            "complexity": complexity,
            "modality_importance": modality_importance,
            "threshold": threshold,
            "visual_importance": modality_importance[:, 0:1],
            "textual_importance": modality_importance[:, 1:2]
        } 