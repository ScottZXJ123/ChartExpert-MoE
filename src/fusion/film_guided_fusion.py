"""
FILM-inspired language-guided visual fusion for ChartExpert-MoE

This module implements language-guided visual feature fusion inspired by FILM (Fusing via 
Instruction Language Models), where textual descriptions guide the fusion of visual information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple
from einops import rearrange


class FILMGuidedFusion(nn.Module):
    """
    Language-guided visual fusion inspired by FILM
    
    Uses language descriptions to modulate and guide visual feature fusion,
    enabling more focused visual processing based on query intent.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.vision_hidden_size = config.get("vision_hidden_size", 768)
        self.text_hidden_size = config.get("text_hidden_size", 4096)
        self.num_heads = config.get("num_heads", 8)
        self.dropout_rate = config.get("dropout_rate", 0.1)
        
        # Language-to-visual conditioning networks
        self.language_to_scale = nn.Sequential(
            nn.Linear(self.text_hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.vision_hidden_size),
            nn.Sigmoid()  # Scale factors
        )
        
        self.language_to_shift = nn.Sequential(
            nn.Linear(self.text_hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.vision_hidden_size),
            nn.Tanh()  # Shift factors
        )
        
        # Multi-head attention for language-visual interaction
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # Gating mechanism for fusion control
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        query_embedding: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply language-guided fusion to visual features
        
        Args:
            visual_features: Visual features [batch_size, num_patches, vision_hidden_size]
            text_features: Text features [batch_size, seq_len, text_hidden_size]
            query_embedding: Optional query-specific embedding
            attention_mask: Attention mask for text
            
        Returns:
            Fused features [batch_size, num_patches, hidden_size]
        """
        batch_size, num_patches, _ = visual_features.shape
        
        # Extract language guidance signals
        if query_embedding is not None:
            language_context = query_embedding
        else:
            # Use mean pooling of text features as context
            if attention_mask is not None:
                language_context = (text_features * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
            else:
                language_context = text_features.mean(dim=1)
        
        # Generate modulation parameters from language
        scale_factors = self.language_to_scale(language_context)  # [batch_size, vision_hidden_size]
        shift_factors = self.language_to_shift(language_context)  # [batch_size, vision_hidden_size]
        
        # Apply FiLM modulation to visual features
        scale_factors = scale_factors.unsqueeze(1).expand(-1, num_patches, -1)
        shift_factors = shift_factors.unsqueeze(1).expand(-1, num_patches, -1)
        
        modulated_visual = visual_features * scale_factors + shift_factors
        
        # Project to common dimension if needed
        if visual_features.size(-1) != self.hidden_size:
            visual_proj = nn.Linear(visual_features.size(-1), self.hidden_size).to(visual_features.device)
            modulated_visual = visual_proj(modulated_visual)
        
        # Apply cross-attention between modulated visual and text features
        attended_features, attention_weights = self.cross_attention(
            query=modulated_visual,
            key=text_features[:, :modulated_visual.size(1), :self.hidden_size],
            value=text_features[:, :modulated_visual.size(1), :self.hidden_size],
            attn_mask=attention_mask[:, :modulated_visual.size(1)] if attention_mask is not None else None
        )
        
        # Gated fusion of original and attended features
        fusion_input = torch.cat([modulated_visual, attended_features], dim=-1)
        gate_values = self.fusion_gate(fusion_input)
        
        fused_features = gate_values * attended_features + (1 - gate_values) * modulated_visual
        
        # Final projection and residual connection
        output = self.output_projection(fused_features)
        output = self.dropout(output)
        
        # Residual connection with original visual features
        if visual_features.size(-1) == self.hidden_size:
            output = output + visual_features
        
        return output
    
    def generate_visual_focus_map(
        self,
        visual_features: torch.Tensor,
        language_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate a focus map indicating which visual regions are most relevant
        
        Args:
            visual_features: Visual features [batch_size, num_patches, vision_hidden_size]
            language_context: Language context [batch_size, text_hidden_size]
            
        Returns:
            Focus map [batch_size, num_patches]
        """
        # Generate scale factors as proxy for importance
        scale_factors = self.language_to_scale(language_context)
        scale_factors = scale_factors.unsqueeze(1).expand(-1, visual_features.size(1), -1)
        
        # Compute importance scores
        importance = (visual_features * scale_factors).mean(dim=-1)
        focus_map = torch.softmax(importance, dim=-1)
        
        return focus_map 