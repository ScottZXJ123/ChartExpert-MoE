"""
Loss functions for ChartExpert-MoE training

Implements specialized loss functions for MoE training including auxiliary losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, List


class ChartMoELoss(nn.Module):
    """
    Combined loss function for ChartExpert-MoE training
    
    Includes:
    - Language modeling loss
    - Auxiliary MoE losses (load balancing, router entropy)
    - Optional task-specific losses
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.aux_loss_weight = config.get("aux_loss_weight", 0.01)
        self.label_smoothing = config.get("label_smoothing", 0.0)
        
        # Language modeling loss
        self.lm_loss = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=self.label_smoothing
        )
        
        # Auxiliary losses
        self.load_balance_loss = LoadBalanceLoss()
        self.router_entropy_loss = RouterEntropyLoss()
        
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        routing_weights: Optional[torch.Tensor] = None,
        aux_loss: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss
        
        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            labels: Target labels [batch_size, seq_len]
            routing_weights: Expert routing weights [batch_size, seq_len, num_experts]
            aux_loss: Pre-computed auxiliary loss from MoE layer
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # Language modeling loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = self.lm_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        losses["lm_loss"] = lm_loss
        
        # Auxiliary MoE losses
        total_aux_loss = torch.tensor(0.0, device=logits.device)
        
        if aux_loss is not None:
            total_aux_loss = total_aux_loss + aux_loss
            losses["moe_aux_loss"] = aux_loss
        
        if routing_weights is not None:
            # Load balance loss
            lb_loss = self.load_balance_loss(routing_weights)
            total_aux_loss = total_aux_loss + lb_loss
            losses["load_balance_loss"] = lb_loss
            
            # Router entropy loss
            entropy_loss = self.router_entropy_loss(routing_weights)
            total_aux_loss = total_aux_loss + entropy_loss
            losses["router_entropy_loss"] = entropy_loss
        
        losses["aux_loss"] = total_aux_loss
        
        # Combined loss
        total_loss = lm_loss + self.aux_loss_weight * total_aux_loss
        losses["loss"] = total_loss
        
        return losses


class LoadBalanceLoss(nn.Module):
    """
    Load balance loss to encourage even distribution of tokens across experts
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate load balance loss
        
        Args:
            routing_weights: [batch_size, seq_len, num_experts]
            
        Returns:
            Load balance loss scalar
        """
        # Calculate mean routing probability per expert
        routing_probs = routing_weights.mean(dim=(0, 1))  # [num_experts]
        
        # Calculate tokens per expert
        tokens_per_expert = routing_weights.sum(dim=(0, 1))  # [num_experts]
        tokens_per_expert = tokens_per_expert / (routing_weights.sum() + self.eps)
        
        # Load balance loss encourages uniform distribution
        num_experts = routing_weights.size(-1)
        lb_loss = num_experts * torch.sum(routing_probs * tokens_per_expert)
        
        return lb_loss


class RouterEntropyLoss(nn.Module):
    """
    Router entropy loss to prevent routing collapse
    """
    
    def __init__(self, target_entropy: Optional[float] = None):
        super().__init__()
        self.target_entropy = target_entropy
    
    def forward(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate router entropy loss
        
        Args:
            routing_weights: [batch_size, seq_len, num_experts]
            
        Returns:
            Entropy loss scalar
        """
        # Calculate entropy of routing distribution
        routing_probs = routing_weights + 1e-8  # Add small epsilon for stability
        entropy = -torch.sum(routing_probs * torch.log(routing_probs), dim=-1)
        mean_entropy = entropy.mean()
        
        if self.target_entropy is not None:
            # Encourage specific entropy level
            entropy_loss = F.mse_loss(mean_entropy, torch.tensor(self.target_entropy))
        else:
            # Maximize entropy (negative because we minimize loss)
            entropy_loss = -mean_entropy
        
        return entropy_loss


class AuxiliaryLoss(nn.Module):
    """
    Additional auxiliary losses for ChartExpert-MoE
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Expert diversity loss
        self.diversity_weight = config.get("diversity_weight", 0.01)
        
        # Consistency loss for cross-modal alignment
        self.consistency_weight = config.get("consistency_weight", 0.1)
        
    def expert_diversity_loss(
        self,
        expert_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Encourage diversity among expert outputs
        
        Args:
            expert_outputs: List of expert output tensors
            
        Returns:
            Diversity loss scalar
        """
        if len(expert_outputs) < 2:
            return torch.tensor(0.0)
        
        # Calculate pairwise cosine similarity
        diversity_loss = 0.0
        num_pairs = 0
        
        for i in range(len(expert_outputs)):
            for j in range(i + 1, len(expert_outputs)):
                # Flatten and normalize
                output_i = F.normalize(expert_outputs[i].view(expert_outputs[i].size(0), -1), dim=1)
                output_j = F.normalize(expert_outputs[j].view(expert_outputs[j].size(0), -1), dim=1)
                
                # Cosine similarity
                similarity = torch.sum(output_i * output_j, dim=1).mean()
                diversity_loss += similarity
                num_pairs += 1
        
        # We want to minimize similarity (maximize diversity)
        diversity_loss = diversity_loss / max(num_pairs, 1)
        
        return diversity_loss * self.diversity_weight
    
    def cross_modal_consistency_loss(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        alignment_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Ensure consistency between visual and textual representations
        
        Args:
            visual_features: Visual feature representations
            text_features: Text feature representations
            alignment_scores: Optional alignment scores from model
            
        Returns:
            Consistency loss scalar
        """
        # Normalize features
        visual_norm = F.normalize(visual_features, dim=-1)
        text_norm = F.normalize(text_features, dim=-1)
        
        # Calculate similarity matrix
        similarity = torch.matmul(visual_norm, text_norm.transpose(-2, -1))
        
        # Create target (diagonal should be high, off-diagonal low)
        batch_size = similarity.size(0)
        target = torch.eye(similarity.size(-1), device=similarity.device)
        target = target.unsqueeze(0).expand(batch_size, -1, -1)
        
        # MSE loss between similarity and target
        consistency_loss = F.mse_loss(similarity, target)
        
        return consistency_loss * self.consistency_weight 