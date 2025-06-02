"""
Mixture of Experts (MoE) layer implementation for ChartExpert-MoE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import math
from .base_models import VisionEncoder, LLMBackbone


class LoadBalancingLoss(nn.Module):
    """
    Load balancing loss for MoE to encourage even distribution across experts
    """
    def __init__(self, num_experts: int = 12, eps: float = 1e-8):
        super().__init__()
        self.num_experts = num_experts
        self.eps = eps
        
    def forward(
        self,
        routing_weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute load balancing loss
        
        Args:
            routing_weights: [batch_size, seq_len, num_experts]
            attention_mask: Optional mask [batch_size, seq_len]
            
        Returns:
            Load balancing loss scalar
        """
        if attention_mask is not None:
            routing_weights = routing_weights * attention_mask.unsqueeze(-1)
            
        # Mean routing probability per expert
        routing_probs = routing_weights.mean(dim=(0, 1))  # [num_experts]
        
        # Tokens per expert
        tokens_per_expert = routing_weights.sum(dim=(0, 1))  # [num_experts]
        tokens_per_expert = tokens_per_expert / (routing_weights.sum() + self.eps)
        
        # Load balance loss encourages uniform distribution
        lb_loss = self.num_experts * torch.sum(routing_probs * tokens_per_expert)
        
        return lb_loss


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with advanced routing capabilities
    
    Features:
    - Top-k gating with load balancing
    - Noisy gating for exploration during training
    - Capacity factor and token dropping
    - Batch priority routing (BPR)
    - Auxiliary losses for training stability
    """
    
    def __init__(
        self,
        experts: List[nn.Module],
        router: nn.Module,
        config: Dict[str, Any]
    ):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.router = router
        self.num_experts = len(experts)
        self.top_k = config.get("top_k", 2)
        self.capacity_factor = config.get("capacity_factor", 1.25)
        self.aux_loss_weight = config.get("aux_loss_weight", 0.01)
        self.load_balancing = config.get("load_balancing", True)
        
        # Noisy gating parameters
        self.noisy_gating = config.get("noisy_gating", False)
        self.noise_eps = config.get("noise_eps", 0.01)
        
        # Batch priority routing
        self.batch_priority_routing = config.get("batch_priority_routing", False)
        
        # Initialize load balancing loss
        self.load_balancing_loss = LoadBalancingLoss()
        
        # MoE configuration
        self.hidden_size = config.get("hidden_size", 768)
        
        # Load balancing components
        if self.load_balancing:
            self.expert_usage = torch.zeros(self.num_experts)
            self.total_tokens = 0
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MoE layer
        
        Args:
            hidden_states: Input features [batch_size, seq_len, hidden_size]
            image: Optional image tensor for context
            input_ids: Optional input token ids for context
            attention_mask: Optional attention mask
            
        Returns:
            Dictionary containing expert outputs, routing weights, and auxiliary loss
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Flatten for processing
        flat_hidden_states = hidden_states.view(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
        
        # Get routing decisions from router
        routing_output = self.router(
            hidden_states=flat_hidden_states,
            image=image,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        router_logits = routing_output["logits"]  # [batch_size * seq_len, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1)  # [batch_size * seq_len, num_experts]
        
        # Apply top-k gating
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # Create dispatch mask
        dispatch_mask = torch.zeros_like(routing_weights)
        dispatch_mask.scatter_(1, top_k_indices, top_k_gates)
        
        # Compute expert outputs
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # Get tokens assigned to this expert
            expert_mask = dispatch_mask[:, i]  # [batch_size * seq_len]
            expert_tokens = flat_hidden_states * expert_mask.unsqueeze(-1)  # [batch_size * seq_len, hidden_size]
            
            # Process through expert
            if torch.sum(expert_mask) > 0:  # Only process if tokens are assigned
                expert_output = expert(
                    hidden_states=expert_tokens,
                    image=image,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    routing_weights=expert_mask
                )
                
                if isinstance(expert_output, dict):
                    expert_output = expert_output.get("hidden_states", expert_tokens)
                
                expert_outputs.append(expert_output)
            else:
                # No tokens assigned to this expert
                expert_outputs.append(torch.zeros_like(expert_tokens))
        
        # Combine expert outputs
        combined_output = torch.zeros_like(flat_hidden_states)
        for i, expert_output in enumerate(expert_outputs):
            expert_weight = dispatch_mask[:, i].unsqueeze(-1)  # [batch_size * seq_len, 1]
            combined_output += expert_weight * expert_output
        
        # Reshape back to original shape
        final_output = combined_output.view(batch_size, seq_len, hidden_size)
        
        # Calculate auxiliary loss for load balancing
        aux_loss = self._calculate_aux_loss(routing_weights, dispatch_mask)
        
        # Update expert usage statistics
        if self.training and self.load_balancing:
            self._update_expert_usage(dispatch_mask.sum(dim=0))
        
        return {
            "expert_outputs": final_output,
            "routing_weights": routing_weights.view(batch_size, seq_len, self.num_experts),
            "dispatch_mask": dispatch_mask.view(batch_size, seq_len, self.num_experts),
            "aux_loss": aux_loss,
            "router_logits": router_logits.view(batch_size, seq_len, self.num_experts)
        }
    
    def _calculate_aux_loss(
        self,
        routing_weights: torch.Tensor,
        dispatch_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate auxiliary loss for load balancing
        
        Args:
            routing_weights: Routing probabilities [batch_size * seq_len, num_experts]
            dispatch_mask: Dispatch decisions [batch_size * seq_len, num_experts]
            
        Returns:
            Auxiliary loss scalar
        """
        if not self.load_balancing:
            return torch.tensor(0.0, device=routing_weights.device)
        
        # Calculate expert utilization
        num_tokens = routing_weights.size(0)
        
        # Mean routing probability for each expert
        mean_routing_prob = torch.mean(routing_weights, dim=0)  # [num_experts]
        
        # Fraction of tokens assigned to each expert
        tokens_per_expert = torch.sum(dispatch_mask, dim=0) / num_tokens  # [num_experts]
        
        # Auxiliary loss: encourage balanced load
        aux_loss = torch.sum(mean_routing_prob * tokens_per_expert) * self.num_experts
        
        return aux_loss
    
    def _update_expert_usage(self, expert_counts: torch.Tensor):
        """Update expert usage statistics"""
        self.expert_usage += expert_counts.cpu()
        self.total_tokens += torch.sum(expert_counts).cpu()
    
    def get_expert_usage_stats(self) -> Dict[str, float]:
        """Get expert usage statistics"""
        if self.total_tokens == 0:
            return {f"expert_{i}": 0.0 for i in range(self.num_experts)}
        
        usage_percentages = (self.expert_usage / self.total_tokens) * 100
        return {f"expert_{i}": float(usage_percentages[i]) for i in range(self.num_experts)}
    
    def reset_usage_stats(self):
        """Reset expert usage statistics"""
        self.expert_usage = torch.zeros(self.num_experts)
        self.total_tokens = 0

    def _add_noise_to_gates(self, gates: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Add noise to gates for exploration during training"""
        if self.noisy_gating and training:
            noise = torch.randn_like(gates) * self.noise_eps
            gates = gates + noise
        return gates
    
    def _compute_capacity(self, batch_size: int, seq_len: int) -> int:
        """Compute expert capacity based on capacity factor"""
        tokens_per_expert = (batch_size * seq_len) / self.num_experts
        capacity = int(self.capacity_factor * tokens_per_expert)
        return capacity
    
    def _batch_priority_routing(
        self,
        routing_weights: torch.Tensor,
        hidden_states: torch.Tensor,
        importance_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implement batch priority routing (BPR)
        Prioritize important tokens when routing to experts
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        if importance_scores is None:
            # Use L2 norm as default importance score
            importance_scores = torch.norm(hidden_states, dim=-1)
        
        # Flatten for processing
        importance_flat = importance_scores.view(-1)
        routing_flat = routing_weights.view(-1, self.num_experts)
        
        # Sort by importance
        sorted_importance, sorted_indices = torch.sort(
            importance_flat, descending=True
        )
        
        # Apply routing with priority
        capacity = self._compute_capacity(batch_size, seq_len)
        expert_counts = torch.zeros(self.num_experts, device=hidden_states.device)
        
        final_routing = torch.zeros_like(routing_flat)
        
        for idx in sorted_indices:
            token_routing = routing_flat[idx]
            top_experts = torch.topk(token_routing, self.top_k).indices
            
            for expert_idx in top_experts:
                if expert_counts[expert_idx] < capacity:
                    final_routing[idx, expert_idx] = token_routing[expert_idx]
                    expert_counts[expert_idx] += 1
        
        # Reshape back
        final_routing = final_routing.view(batch_size, seq_len, self.num_experts)
        
        return final_routing, expert_counts


class SparseDispatcher:
    """
    Helper class for efficiently dispatching tokens to experts
    """
    
    def __init__(
        self,
        num_experts: int,
        gates: torch.Tensor,
        capacity_factor: float = 1.25
    ):
        self.num_experts = num_experts
        self.gates = gates
        self.capacity_factor = capacity_factor
        
        # Calculate capacity per expert
        batch_size, seq_len = gates.shape[:2]
        self.capacity = int(capacity_factor * batch_size * seq_len / num_experts)
        
        self._create_dispatch_plan()
    
    def _create_dispatch_plan(self):
        """Create dispatch plan for routing tokens to experts"""
        # Get top expert for each token
        self.expert_indices = torch.argmax(self.gates, dim=-1)  # [batch_size, seq_len]
        
        # Create dispatch mask with capacity constraints
        self.dispatch_mask = torch.zeros_like(self.gates)
        
        for expert_id in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (self.expert_indices == expert_id)
            expert_positions = torch.nonzero(expert_mask, as_tuple=False)
            
            # Apply capacity constraint
            if len(expert_positions) > self.capacity:
                # Randomly select tokens up to capacity
                perm = torch.randperm(len(expert_positions))[:self.capacity]
                expert_positions = expert_positions[perm]
            
            # Set dispatch mask
            for pos in expert_positions:
                self.dispatch_mask[pos[0], pos[1], expert_id] = 1.0
    
    def dispatch(self, tokens: torch.Tensor) -> List[torch.Tensor]:
        """Dispatch tokens to experts"""
        expert_inputs = []
        
        for expert_id in range(self.num_experts):
            # Get tokens for this expert
            expert_mask = self.dispatch_mask[:, :, expert_id]
            expert_tokens = tokens * expert_mask.unsqueeze(-1)
            expert_inputs.append(expert_tokens)
        
        return expert_inputs
    
    def combine(self, expert_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Combine expert outputs"""
        combined = torch.zeros_like(expert_outputs[0])
        
        for expert_id, expert_output in enumerate(expert_outputs):
            expert_mask = self.dispatch_mask[:, :, expert_id].unsqueeze(-1)
            combined += expert_mask * expert_output
        
        return combined


class NoisyTopKGating(nn.Module):
    """
    Noisy top-k gating mechanism for MoE
    """
    
    def __init__(
        self,
        input_size: int,
        num_experts: int,
        top_k: int = 2,
        noise_epsilon: float = 1e-2
    ):
        super().__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_epsilon = noise_epsilon
        
        # Gating network
        self.gate = nn.Linear(input_size, num_experts, bias=False)
        self.noise_gate = nn.Linear(input_size, num_experts, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize gating weights"""
        nn.init.normal_(self.gate.weight, std=0.1)
        nn.init.normal_(self.noise_gate.weight, std=0.1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of noisy top-k gating
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            Tuple of (gates, load) where gates are the gating weights
            and load is for load balancing
        """
        # Calculate clean logits
        clean_logits = self.gate(x)
        
        if self.training:
            # Add noise during training
            noise_logits = self.noise_gate(x)
            noise = torch.randn_like(clean_logits) * F.softplus(noise_logits) * self.noise_epsilon
            noisy_logits = clean_logits + noise
        else:
            noisy_logits = clean_logits
        
        # Apply top-k selection
        top_k_logits, top_k_indices = torch.topk(noisy_logits, self.top_k, dim=-1)
        
        # Create sparse gates
        gates = torch.zeros_like(noisy_logits)
        gates.scatter_(1, top_k_indices, F.softmax(top_k_logits, dim=-1))
        
        # Calculate load for balancing
        load = F.softmax(clean_logits, dim=-1)
        
        return gates, load 