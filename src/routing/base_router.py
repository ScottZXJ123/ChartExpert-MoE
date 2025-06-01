"""
Base router class for ChartExpert-MoE
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod


class BaseRouter(nn.Module, ABC):
    """
    Abstract base class for routing mechanisms
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_size = config.get("hidden_size", 768)
        self.num_experts = config.get("num_experts", 12)
        self.top_k = config.get("top_k", 2)
        
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route inputs to experts
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Optional attention mask
            
        Returns:
            routing_weights: Weights for each expert
            selected_experts: Indices of selected experts
        """
        pass
    
    def compute_load_balancing_loss(
        self,
        routing_weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute load balancing loss to encourage even expert usage"""
        if attention_mask is not None:
            routing_weights = routing_weights * attention_mask.unsqueeze(-1)
            
        # Average routing probability per expert
        routing_probs = routing_weights.mean(dim=(0, 1))
        
        # Compute load balancing loss
        num_experts = routing_weights.size(-1)
        target_load = 1.0 / num_experts
        load_balancing_loss = num_experts * torch.sum(
            routing_probs * torch.abs(routing_probs - target_load)
        )
        
        return load_balancing_loss 