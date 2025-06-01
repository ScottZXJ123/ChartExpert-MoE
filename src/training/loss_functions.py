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
    
    Combines:
    - Language modeling loss (for text generation)
    - Auxiliary load balancing loss (for MoE routing)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.aux_loss_weight = config.get("aux_loss_weight", 0.01)
        
        # Cross entropy loss for language modeling
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        aux_loss: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            logits: Language model logits [batch_size, seq_len, vocab_size]
            labels: Target token ids [batch_size, seq_len]
            aux_loss: Auxiliary MoE load balancing loss
            
        Returns:
            Dictionary with loss components
        """
        # Language modeling loss
        if logits.dim() == 3:
            # Flatten for cross entropy
            lm_loss = self.lm_criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        else:
            lm_loss = self.lm_criterion(logits, labels)
        
        # Total loss starts with LM loss
        total_loss = lm_loss
        
        # Add auxiliary loss if provided
        if aux_loss is not None:
            total_loss = total_loss + self.aux_loss_weight * aux_loss
        
        return {
            "loss": total_loss,
            "lm_loss": lm_loss,
            "aux_loss": aux_loss if aux_loss is not None else torch.tensor(0.0, device=lm_loss.device)
        }


class AuxiliaryLoss(nn.Module):
    """Auxiliary loss for load balancing in MoE"""
    
    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
    
    def forward(self, gate_logits: torch.Tensor, expert_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss
        
        Args:
            gate_logits: Gating logits [batch_size, num_experts]
            expert_mask: Binary mask of selected experts [batch_size, num_experts]
            
        Returns:
            Auxiliary loss scalar
        """
        # Compute gate probabilities
        gate_probs = torch.softmax(gate_logits, dim=-1)
        
        # Average gate probability per expert
        avg_gate_probs = gate_probs.mean(dim=0)  # [num_experts]
        
        # Average expert selection frequency  
        avg_expert_freq = expert_mask.float().mean(dim=0)  # [num_experts]
        
        # Load balancing loss - encourage uniform distribution
        aux_loss = (avg_gate_probs * avg_expert_freq).sum() * self.num_experts / self.top_k
        
        return aux_loss


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


class ExpertDiversityLoss(nn.Module):
    """
    Expert diversity loss to encourage diversity among expert outputs
    """
    
    def __init__(self, diversity_weight: float = 0.01):
        super().__init__()
        self.diversity_weight = diversity_weight
    
    def forward(
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


class CrossModalConsistencyLoss(nn.Module):
    """
    Cross-modal consistency loss to ensure consistency between visual and textual representations
    """
    
    def __init__(self, consistency_weight: float = 0.1):
        super().__init__()
        self.consistency_weight = consistency_weight
    
    def forward(
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