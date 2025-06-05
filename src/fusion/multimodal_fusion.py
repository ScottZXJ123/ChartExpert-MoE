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
    """Optimized attention-based fusion strategy with adaptive complexity"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = config.get("hidden_size", 768)
        
        # Traditional attention components
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
        
        # Adaptive fusion components
        self.complexity_estimator = nn.Linear(self.hidden_size * 2, 1)
        self.simple_fusion_projection = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Lightweight fusion components
        self.reduction_layer = nn.Linear(self.hidden_size, self.hidden_size // 4)
        self.lightweight_query_proj = nn.Linear(self.hidden_size // 4, self.hidden_size // 4)
        self.lightweight_kv_proj = nn.Linear(self.hidden_size // 4, self.hidden_size // 4)
        self.lightweight_fusion_proj = nn.Linear(self.hidden_size + self.hidden_size // 4, self.hidden_size)
        
        # Sparse attention components
        self.sparse_query_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.sparse_key_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.sparse_value_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Efficient fusion layers using grouped convolution
        self.efficient_fusion_layers = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1, groups=8),
            nn.GELU(),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=1)
        )
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply adaptive attention-based fusion"""
        batch_size = visual_features.size(0)
        visual_seq_len = visual_features.size(1)
        text_seq_len = text_features.size(1)
        
        # 1. Estimate fusion complexity
        complexity_score = self._estimate_fusion_complexity(visual_features, text_features)
        
        # 2. Adaptive fusion strategy based on complexity
        if complexity_score < 0.3:
            # Simple concatenation for low complexity
            concatenated = torch.cat([visual_features, text_features], dim=1)
            return self.simple_fusion_projection(concatenated)
        
        elif complexity_score < 0.7:
            # Lightweight attention for medium complexity
            return self._lightweight_attention_fusion(visual_features, text_features, attention_mask)
        
        else:
            # Full attention for high complexity with sparse optimization
            return self._sparse_attention_fusion(visual_features, text_features, attention_mask)
    
    def _estimate_fusion_complexity(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> float:
        """Estimate fusion complexity based on feature characteristics"""
        # 1. Sequence length complexity
        seq_complexity = min(1.0, (visual_features.size(1) + text_features.size(1)) / 512)
        
        # 2. Feature variance (higher variance = more complex)
        visual_var = visual_features.var(dim=[1, 2]).mean().item()
        text_var = text_features.var(dim=[1, 2]).mean().item()
        var_complexity = min(1.0, (visual_var + text_var) * 10)
        
        # 3. Cross-modal similarity (lower similarity = higher complexity)
        visual_mean = visual_features.mean(dim=1)  # [B, H]
        text_mean = text_features.mean(dim=1)      # [B, H]
        similarity = F.cosine_similarity(visual_mean, text_mean, dim=-1).mean().item()
        similarity_complexity = 1.0 - abs(similarity)
        
        # Combine metrics
        complexity = (seq_complexity + var_complexity + similarity_complexity) / 3
        return min(1.0, complexity)
    
    def _lightweight_attention_fusion(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Lightweight attention fusion for medium complexity"""
        # Use only single-head attention with reduced computation
        batch_size = visual_features.size(0)
        
        # Reduce dimensions for efficiency
        visual_reduced = self.reduction_layer(visual_features)  # [B, V, H/4]
        text_reduced = self.reduction_layer(text_features)      # [B, T, H/4]
        
        # Simple cross attention
        visual_query = self.lightweight_query_proj(visual_reduced)
        text_key_value = self.lightweight_kv_proj(text_reduced)
        
        # Efficient attention computation
        attn_weights = torch.bmm(visual_query, text_key_value.transpose(1, 2))
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        visual_attended = torch.bmm(attn_weights, text_reduced)
        
        # Combine and project back
        combined = torch.cat([visual_features, visual_attended], dim=-1)
        fused = self.lightweight_fusion_proj(combined)
        
        return torch.cat([fused, text_features], dim=1)
    
    def _sparse_attention_fusion(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Sparse attention fusion optimized for chart structure"""
        batch_size = visual_features.size(0)
        visual_seq_len = visual_features.size(1)
        text_seq_len = text_features.size(1)
        
        # Create sparse attention patterns for chart understanding
        sparse_pattern = self._create_chart_sparse_pattern(visual_seq_len, text_seq_len, visual_features.device)
        
        # Apply sparse cross-modal attention
        # Text to Visual (focus on chart regions mentioned in text)
        text_to_visual_attn = self._sparse_cross_attention(
            query=text_features,
            key=visual_features,
            value=visual_features,
            sparse_mask=sparse_pattern['text_to_visual']
        )
        
        # Visual to Text (focus on relevant text for visual regions)
        visual_to_text_attn = self._sparse_cross_attention(
            query=visual_features,
            key=text_features,
            value=text_features,
            sparse_mask=sparse_pattern['visual_to_text']
        )
        
        # Residual connections
        visual_attended = visual_features + visual_to_text_attn
        text_attended = text_features + text_to_visual_attn
        
        # Efficient fusion using grouped convolution
        concatenated = torch.cat([visual_attended, text_attended], dim=1)
        
        # Transpose for conv1d: [B, S, H] -> [B, H, S]
        concatenated_transposed = concatenated.transpose(1, 2)
        fused_transposed = self.efficient_fusion_layers(concatenated_transposed)
        fused = fused_transposed.transpose(1, 2)  # [B, H, S] -> [B, S, H]
        
        return fused
    
    def _create_chart_sparse_pattern(self, visual_len: int, text_len: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Create sparse attention patterns optimized for chart understanding"""
        # Pattern 1: Text to Visual - focus on spatial regions
        text_to_visual = torch.zeros(text_len, visual_len, device=device)
        
        # First few text tokens (usually chart type) attend to all visual
        text_to_visual[:min(5, text_len), :] = 1
        
        # Middle text tokens attend to central visual regions
        mid_start = text_len // 3
        mid_end = 2 * text_len // 3
        visual_center_start = visual_len // 4
        visual_center_end = 3 * visual_len // 4
        text_to_visual[mid_start:mid_end, visual_center_start:visual_center_end] = 1
        
        # Pattern 2: Visual to Text - local and global attention
        visual_to_text = torch.zeros(visual_len, text_len, device=device)
        
        # All visual tokens attend to first few text tokens (chart type info)
        visual_to_text[:, :min(5, text_len)] = 1
        
        # Local attention: visual patches attend to relevant text regions
        window_size = min(8, text_len // 4)
        for i in range(visual_len):
            text_start = min(i * text_len // visual_len, text_len - window_size)
            text_end = min(text_start + window_size, text_len)
            visual_to_text[i, text_start:text_end] = 1
        
        return {
            'text_to_visual': text_to_visual,
            'visual_to_text': visual_to_text
        }
    
    def _sparse_cross_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sparse_mask: torch.Tensor
    ) -> torch.Tensor:
        """Efficient sparse cross-attention computation"""
        batch_size, query_len, hidden_size = query.shape
        key_len = key.size(1)
        
        # Project to attention space
        q = self.sparse_query_proj(query)  # [B, query_len, hidden_size]
        k = self.sparse_key_proj(key)      # [B, key_len, hidden_size]
        v = self.sparse_value_proj(value)  # [B, key_len, hidden_size]
        
        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2))  # [B, query_len, key_len]
        
        # Apply sparse mask
        scores = scores.masked_fill(sparse_mask.unsqueeze(0) == 0, float('-inf'))
        
        # Attention weights and output
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.bmm(attn_weights, v)
        
        return attn_output


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