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
    """Chart-aware optimized attention-based fusion strategy with adaptive complexity"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = config.get("hidden_size", 768)
        self.num_heads = config.get("num_heads", 8)
        self.head_dim = self.hidden_size // self.num_heads
        
        # Chart-aware attention patterns
        self.chart_attention_patterns = {
            'bar_chart': self._create_grid_attention_mask,
            'line_chart': self._create_sequential_attention_mask,
            'pie_chart': self._create_radial_attention_mask,
            'scatter_plot': self._create_local_attention_mask,
            'heatmap': self._create_block_attention_mask
        }
        
        # Flash Attention support
        self.use_flash_attn = config.get("use_flash_attention", True)
        try:
            if self.use_flash_attn:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
        except ImportError:
            self.use_flash_attn = False
            self.flash_attn_func = None
        
        # Traditional attention components
        self.visual_attention = self._create_chart_aware_attention()
        
        self.text_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            batch_first=True
        )
        
        self.fusion_projection = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        # Chart-type classifier for dynamic attention patterns
        self.chart_type_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 5),  # 5 chart types
            nn.Softmax(dim=-1)
        )
        
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
    
    def _create_chart_aware_attention(self):
        """Create chart-aware attention module"""
        return ChartAwareAttention(self.hidden_size, self.num_heads)
    
    def _create_grid_attention_mask(self, seq_len: int, key_regions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Create grid-based attention bias for bar charts"""
        # Create attention bias that favors grid-aligned elements
        mask = torch.zeros(seq_len, seq_len)
        
        # Encourage attention between elements in same row/column
        grid_size = int(seq_len ** 0.5)
        for i in range(seq_len):
            row, col = i // grid_size, i % grid_size
            # Same row attention
            for j in range(row * grid_size, (row + 1) * grid_size):
                if j < seq_len:
                    mask[i, j] = 0.1
            # Same column attention  
            for j in range(col, seq_len, grid_size):
                mask[i, j] = 0.1
        
        return mask
    
    def _create_sequential_attention_mask(self, seq_len: int, key_regions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Create sequential attention bias for line charts"""
        # Encourage attention to neighboring elements and trends
        mask = torch.zeros(seq_len, seq_len)
        
        for i in range(seq_len):
            # Local neighborhood attention
            for j in range(max(0, i-3), min(seq_len, i+4)):
                distance = abs(i - j)
                mask[i, j] = 0.2 / (distance + 1)
        
        return mask
    
    def _create_radial_attention_mask(self, seq_len: int, key_regions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Create radial attention bias for pie charts"""
        # Encourage attention to center and adjacent sectors
        mask = torch.zeros(seq_len, seq_len)
        center_idx = seq_len // 2
        
        for i in range(seq_len):
            # Attention to center
            mask[i, center_idx] = 0.2
            # Attention to adjacent elements (circular)
            left = (i - 1) % seq_len
            right = (i + 1) % seq_len
            mask[i, left] = 0.1
            mask[i, right] = 0.1
        
        return mask
    
    def _create_local_attention_mask(self, seq_len: int, key_regions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Create local attention bias for scatter plots"""
        # Encourage local clustering attention
        mask = torch.zeros(seq_len, seq_len)
        
        # Create local attention windows
        window_size = max(3, seq_len // 10)
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            for j in range(start, end):
                distance = abs(i - j)
                mask[i, j] = 0.15 / (distance + 1)
        
        return mask
    
    def _create_block_attention_mask(self, seq_len: int, key_regions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Create block attention bias for heatmaps"""
        # Encourage block-wise attention patterns
        mask = torch.zeros(seq_len, seq_len)
        
        block_size = max(4, seq_len // 8)
        for i in range(seq_len):
            block_start = (i // block_size) * block_size
            block_end = min(seq_len, block_start + block_size)
            for j in range(block_start, block_end):
                mask[i, j] = 0.1
        
        return mask


class ChartAwareAttention(nn.Module):
    """Chart-aware attention mechanism with Flash Attention optimization"""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Chart pattern detection
        self.pattern_detector = nn.Linear(hidden_size, 5)  # 5 chart types
        
        # Flash Attention support
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
            self.use_flash_attn = True
        except ImportError:
            self.flash_attn_func = None
            self.use_flash_attn = False
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        chart_type: Optional[str] = None,
        key_regions: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Chart-aware attention forward pass
        
        Args:
            query: Query tensor [batch_size, seq_len, hidden_size]
            key: Key tensor [batch_size, seq_len, hidden_size]  
            value: Value tensor [batch_size, seq_len, hidden_size]
            chart_type: Chart type hint
            key_regions: Important regions in the chart
            key_padding_mask: Padding mask
            
        Returns:
            Attention output and weights
        """
        batch_size, seq_len = query.shape[:2]
        
        # Project to q, k, v
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.use_flash_attn and query.is_cuda:
            # Use Flash Attention for efficiency
            if chart_type and hasattr(self, '_get_chart_bias'):
                attention_bias = self._get_chart_bias(chart_type, seq_len, query.device)
                attn_output = self.flash_attn_func(q, k, v, bias=attention_bias)
            else:
                attn_output = self.flash_attn_func(q, k, v)
            
            # No attention weights returned from Flash Attention
            attn_weights = None
        else:
            # Standard attention with chart-specific patterns
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            # Apply chart-specific attention bias
            if chart_type:
                chart_bias = self._get_chart_bias(chart_type, seq_len, query.device)
                if chart_bias is not None:
                    scores = scores + chart_bias.unsqueeze(0).unsqueeze(0)  # Broadcast for batch and heads
            
            # Apply padding mask
            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attn_output)
        
        return output, attn_weights
    
    def _get_chart_bias(self, chart_type: str, seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
        """Get attention bias for specific chart type"""
        if chart_type == 'bar_chart':
            return self._create_grid_bias(seq_len, device)
        elif chart_type == 'line_chart':
            return self._create_sequential_bias(seq_len, device)
        elif chart_type == 'pie_chart':
            return self._create_radial_bias(seq_len, device)
        elif chart_type == 'scatter_plot':
            return self._create_local_bias(seq_len, device)
        elif chart_type == 'heatmap':
            return self._create_block_bias(seq_len, device)
        else:
            return None
    
    def _create_grid_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Grid attention bias for bar charts"""
        bias = torch.zeros(seq_len, seq_len, device=device)
        grid_size = int(seq_len ** 0.5)
        
        for i in range(seq_len):
            row, col = i // grid_size, i % grid_size
            # Same row/column elements get positive bias
            for j in range(row * grid_size, (row + 1) * grid_size):
                if j < seq_len:
                    bias[i, j] = 0.1
            for j in range(col, seq_len, grid_size):
                bias[i, j] = 0.1
        
        return bias
    
    def _create_sequential_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Sequential attention bias for line charts"""
        bias = torch.zeros(seq_len, seq_len, device=device)
        
        for i in range(seq_len):
            for j in range(max(0, i-2), min(seq_len, i+3)):
                distance = abs(i - j) 
                bias[i, j] = 0.2 / (distance + 1)
        
        return bias
    
    def _create_radial_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Radial attention bias for pie charts"""
        bias = torch.zeros(seq_len, seq_len, device=device)
        center_idx = seq_len // 2
        
        for i in range(seq_len):
            bias[i, center_idx] = 0.2  # Center attention
            # Adjacent sectors
            left = (i - 1) % seq_len
            right = (i + 1) % seq_len
            bias[i, left] = 0.1
            bias[i, right] = 0.1
        
        return bias
    
    def _create_local_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Local attention bias for scatter plots"""
        bias = torch.zeros(seq_len, seq_len, device=device)
        window = max(3, seq_len // 10)
        
        for i in range(seq_len):
            start = max(0, i - window)
            end = min(seq_len, i + window + 1)
            for j in range(start, end):
                distance = abs(i - j)
                bias[i, j] = 0.15 / (distance + 1)
        
        return bias
    
    def _create_block_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Block attention bias for heatmaps"""
        bias = torch.zeros(seq_len, seq_len, device=device)
        block_size = max(4, seq_len // 8)
        
        for i in range(seq_len):
            block_start = (i // block_size) * block_size
            block_end = min(seq_len, block_start + block_size)
            for j in range(block_start, block_end):
                bias[i, j] = 0.1
        
        return bias
    
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