"""
Base expert class for ChartExpert-MoE expert modules
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict


class SharedExpertBackbone(nn.Module):
    """共享专家骨干网络，减少重复计算"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_size = config.get("hidden_size", 768)
        
        # 共享的特征提取器 - 所有专家共用
        self.shared_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.get("dropout_rate", 0.1)),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )
        
        # 共享的注意力机制
        self.shared_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=config.get("dropout_rate", 0.1)
        )
        
        # 缓存机制
        self._feature_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self.max_cache_size = config.get("max_cache_size", 100)
    
    def forward(self, hidden_states: torch.Tensor, use_cache: bool = True) -> torch.Tensor:
        """
        共享特征提取
        
        Args:
            hidden_states: 输入特征 [batch_size * seq_len, hidden_size]
            use_cache: 是否使用缓存
            
        Returns:
            共享特征 [batch_size * seq_len, hidden_size]
        """
        # 生成缓存键
        if use_cache and self.training is False:  # 只在推理时缓存
            cache_key = self._generate_cache_key(hidden_states)
            if cache_key in self._feature_cache:
                self._cache_hits += 1
                return self._feature_cache[cache_key]
            self._cache_misses += 1
        
        # 共享编码
        shared_features = self.shared_encoder(hidden_states)
        
        # 共享自注意力（如果序列足够长）
        if hidden_states.size(0) > 1:
            # 重塑为序列格式进行注意力计算
            seq_len = min(int(hidden_states.size(0) ** 0.5), 32)  # 限制序列长度
            if seq_len > 1:
                batch_size = hidden_states.size(0) // seq_len
                if batch_size * seq_len == hidden_states.size(0):
                    seq_features = shared_features[:batch_size * seq_len].view(batch_size, seq_len, self.hidden_size)
                    attended_features, _ = self.shared_attention(seq_features, seq_features, seq_features)
                    shared_features[:batch_size * seq_len] = attended_features.view(-1, self.hidden_size)
        
        # 缓存结果
        if use_cache and self.training is False:
            # 限制缓存大小
            if len(self._feature_cache) >= self.max_cache_size:
                oldest_key = next(iter(self._feature_cache))
                del self._feature_cache[oldest_key]
            self._feature_cache[cache_key] = shared_features.detach().clone()
        
        return shared_features
    
    def _generate_cache_key(self, tensor: torch.Tensor) -> str:
        """生成张量的缓存键"""
        # 使用张量的统计信息和形状作为键
        with torch.no_grad():
            mean_val = tensor.mean().item()
            std_val = tensor.std().item()
            shape_str = "_".join(map(str, tensor.shape))
            device_str = str(tensor.device)
            return f"{mean_val:.4f}_{std_val:.4f}_{shape_str}_{device_str}"
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._feature_cache),
            "max_cache_size": self.max_cache_size
        }
    
    def clear_cache(self):
        """清空缓存"""
        self._feature_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


class BaseExpert(nn.Module, ABC):
    """
    Base class for all expert modules in ChartExpert-MoE
    
    All expert modules should inherit from this class and implement the forward method.
    This ensures consistent interface and provides common functionality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_size = config.get("hidden_size", 768)
        self.expert_type = config.get("expert_type", "base")
        self.dropout_rate = config.get("dropout_rate", 0.1)
        
        # Common layers that most experts will use
        self.input_norm = nn.LayerNorm(self.hidden_size)
        self.output_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Expert-specific processing layers (to be defined in subclasses)
        self.expert_layers = self._build_expert_layers(config)
        
        # Output projection to ensure consistent output size
        expert_output_size = config.get("expert_hidden_size", self.hidden_size)
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Activation tracking for analysis
        self.activation_count = 0
        self.total_tokens_processed = 0
    
    @abstractmethod
    def _build_expert_layers(self, config: Dict[str, Any]) -> nn.Module:
        """
        Build expert-specific layers. Must be implemented by subclasses.
        
        Args:
            config: Expert configuration
            
        Returns:
            Expert-specific neural network layers
        """
        pass
    
    @abstractmethod
    def _expert_forward(
        self,
        hidden_states: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Expert-specific forward pass. Must be implemented by subclasses.
        
        Args:
            hidden_states: Input hidden states [batch_size * seq_len, hidden_size]
            image: Optional image tensor for visual experts
            input_ids: Optional input token ids for text-aware experts
            attention_mask: Optional attention mask
            
        Returns:
            Processed hidden states [batch_size * seq_len, hidden_size]
        """
        pass
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Common forward pass for all experts
        
        Args:
            hidden_states: Input hidden states [batch_size * seq_len, hidden_size]
            image: Optional image tensor
            input_ids: Optional input token ids
            attention_mask: Optional attention mask
            routing_weights: Optional routing weights for this expert
            
        Returns:
            Dictionary containing processed hidden states and metadata
        """
        # Input validation
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"hidden_states must be a torch.Tensor, got {type(hidden_states)}")
        
        if hidden_states.dim() != 2:
            raise ValueError(f"hidden_states must be 2D tensor [batch_size * seq_len, hidden_size], got shape {hidden_states.shape}")
        
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError(f"hidden_states last dimension must be {self.hidden_size}, got {hidden_states.size(-1)}")
        
        # Check for NaN or Inf values
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            raise ValueError("hidden_states contains NaN or Inf values")
        
        # Track activation
        self.activation_count += 1
        if routing_weights is not None:
            self.total_tokens_processed += torch.sum(routing_weights > 0).item()
        
        # Normalize input
        normalized_input = self.input_norm(hidden_states)
        
        # Apply expert-specific processing
        expert_output = self._expert_forward(
            hidden_states=normalized_input,
            image=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Apply dropout
        expert_output = self.dropout(expert_output)
        
        # Project to output size
        projected_output = self.output_projection(expert_output)
        
        # Apply output normalization
        normalized_output = self.output_norm(projected_output)
        
        # Residual connection
        final_output = normalized_output + hidden_states
        
        return {
            "hidden_states": final_output,
            "expert_type": self.expert_type,
            "activation_strength": routing_weights.sum().item() if routing_weights is not None else 1.0
        }
    
    def get_activation_stats(self) -> Dict[str, float]:
        """Get expert activation statistics"""
        return {
            "activation_count": self.activation_count,
            "total_tokens_processed": self.total_tokens_processed,
            "avg_tokens_per_activation": (
                self.total_tokens_processed / max(1, self.activation_count)
            )
        }
    
    def reset_stats(self):
        """Reset activation statistics"""
        self.activation_count = 0
        self.total_tokens_processed = 0
    
    def get_expert_type(self) -> str:
        """Get the expert type identifier"""
        return self.expert_type
    
    def is_visual_expert(self) -> bool:
        """Check if this is a visual expert that needs image input"""
        return self.expert_type in [
            "layout", "ocr", "scale", "geometric", "trend", "alignment"
        ]
    
    def is_text_expert(self) -> bool:
        """Check if this is a text expert that primarily processes language"""
        return self.expert_type in [
            "query", "numerical", "integration", "orchestrator"
        ]
    
    def get_parameter_count(self) -> int:
        """Get the number of parameters in this expert"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLPExpert(BaseExpert):
    """
    Simple MLP-based expert implementation
    Can be used as a baseline or for experts that don't need specialized architectures
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def _build_expert_layers(self, config: Dict[str, Any]) -> nn.Module:
        """Build MLP layers"""
        hidden_size = self.hidden_size
        expert_hidden_size = config.get("expert_hidden_size", hidden_size * 4)
        num_layers = config.get("num_layers", 2)
        
        layers = []
        input_size = hidden_size
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer
                output_size = hidden_size
            else:
                output_size = expert_hidden_size
            
            layers.extend([
                nn.Linear(input_size, output_size),
                nn.GELU() if i < num_layers - 1 else nn.Identity(),
            ])
            input_size = output_size
        
        return nn.Sequential(*layers)
    
    def _expert_forward(
        self,
        hidden_states: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Simple MLP forward pass"""
        return self.expert_layers(hidden_states)


class AttentionExpert(BaseExpert):
    """
    Attention-based expert that can attend to different parts of the input
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_heads = config.get("num_heads", 8)
        self.head_dim = self.hidden_size // self.num_heads
        
    def _build_expert_layers(self, config: Dict[str, Any]) -> nn.Module:
        """Build multi-head attention layers"""
        return nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
    
    def _expert_forward(
        self,
        hidden_states: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Attention-based forward pass"""
        # Reshape for attention (assuming flat input)
        batch_size = hidden_states.size(0)
        seq_len = 1  # Since input is flattened
        
        # Self-attention
        attended_output, _ = self.expert_layers(
            hidden_states.unsqueeze(1),  # Add sequence dimension
            hidden_states.unsqueeze(1),
            hidden_states.unsqueeze(1)
        )
        
        return attended_output.squeeze(1)  # Remove sequence dimension 