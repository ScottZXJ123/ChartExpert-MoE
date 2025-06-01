"""
Performance optimization utilities for ChartExpert-MoE

Implements various optimization techniques including FlashAttention,
quantization, pruning, and knowledge distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from collections import OrderedDict


class FlashAttentionWrapper(nn.Module):
    """
    Wrapper for FlashAttention to optimize attention computation
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.use_flash_attn = config.get("use_flash_attention", False)
        self.dropout_p = config.get("attention_dropout", 0.0)
        
        if self.use_flash_attn:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
            except ImportError:
                print("FlashAttention not available, falling back to standard attention")
                self.use_flash_attn = False
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply optimized attention"""
        if self.use_flash_attn and q.is_cuda:
            # FlashAttention expects (batch, seqlen, nheads, headdim)
            return self.flash_attn_func(q, k, v, self.dropout_p)
        else:
            # Standard attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
            if attention_mask is not None:
                scores = scores + attention_mask
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
            return torch.matmul(attn_weights, v)


class ModelQuantizer:
    """
    Quantization utilities for model compression
    """
    
    @staticmethod
    def quantize_model(
        model: nn.Module,
        quantization_config: Dict[str, Any]
    ) -> nn.Module:
        """
        Apply quantization to model
        
        Args:
            model: Model to quantize
            quantization_config: Quantization configuration
            
        Returns:
            Quantized model
        """
        quantization_type = quantization_config.get("type", "dynamic")
        
        if quantization_type == "dynamic":
            # Dynamic quantization (most compatible)
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                qconfig_spec={
                    nn.Linear: torch.quantization.default_dynamic_qconfig,
                    nn.LSTM: torch.quantization.default_dynamic_qconfig,
                },
                dtype=torch.qint8
            )
        elif quantization_type == "static":
            # Static quantization (requires calibration)
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            # Note: Calibration step would go here
            quantized_model = torch.quantization.convert(model, inplace=False)
        elif quantization_type == "qat":
            # Quantization-aware training
            model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            quantized_model = torch.quantization.prepare_qat(model, inplace=False)
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
        
        return quantized_model
    
    @staticmethod
    def int8_quantize_experts(
        experts: List[nn.Module],
        calibration_data: Optional[torch.Tensor] = None
    ) -> List[nn.Module]:
        """Quantize expert modules to INT8"""
        quantized_experts = []
        
        for expert in experts:
            # Apply INT8 quantization to each expert
            expert_int8 = torch.quantization.quantize_dynamic(
                expert,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            quantized_experts.append(expert_int8)
        
        return quantized_experts


class ModelPruner:
    """
    Pruning utilities for model compression
    """
    
    @staticmethod
    def structured_pruning(
        model: nn.Module,
        pruning_config: Dict[str, Any]
    ) -> nn.Module:
        """
        Apply structured pruning to model
        
        Args:
            model: Model to prune
            pruning_config: Pruning configuration
            
        Returns:
            Pruned model
        """
        import torch.nn.utils.prune as prune
        
        pruning_ratio = pruning_config.get("ratio", 0.1)
        pruning_norm = pruning_config.get("norm", 2)
        
        # Get all Linear and Conv2d modules
        modules_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                modules_to_prune.append((module, 'weight'))
        
        # Apply structured pruning
        prune.global_unstructured(
            modules_to_prune,
            pruning_method=prune.L2Unstructured,
            amount=pruning_ratio,
        )
        
        # Remove pruning reparameterization to make it permanent
        for module, param_name in modules_to_prune:
            prune.remove(module, param_name)
        
        return model
    
    @staticmethod
    def prune_redundant_experts(
        model: nn.Module,
        expert_importance_scores: Dict[str, float],
        threshold: float = 0.1
    ) -> nn.Module:
        """Prune experts with low importance scores"""
        # Identify experts to remove
        experts_to_remove = [
            name for name, score in expert_importance_scores.items()
            if score < threshold
        ]
        
        # Remove low-importance experts
        for expert_name in experts_to_remove:
            if hasattr(model, expert_name):
                delattr(model, expert_name)
                print(f"Pruned expert: {expert_name}")
        
        # Update expert list
        if hasattr(model, 'experts'):
            model.experts = nn.ModuleList([
                expert for i, expert in enumerate(model.experts)
                if f"expert_{i}" not in experts_to_remove
            ])
        
        return model


class KnowledgeDistiller:
    """
    Knowledge distillation utilities
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        temperature: float = 3.0,
        alpha: float = 0.7
    ):
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def distillation_loss(
        self,
        student_outputs: torch.Tensor,
        labels: torch.Tensor,
        student_hidden_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute distillation loss
        
        Args:
            student_outputs: Student model outputs
            labels: Ground truth labels
            student_hidden_states: Optional hidden states for feature distillation
            
        Returns:
            Combined distillation loss
        """
        # Get teacher outputs
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                student_outputs.detach()
            )
        
        # Soft target loss
        soft_loss = F.kl_div(
            F.log_softmax(student_outputs / self.temperature, dim=-1),
            F.softmax(teacher_outputs / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard target loss
        hard_loss = F.cross_entropy(student_outputs, labels)
        
        # Combined loss
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return loss


class BatchInferenceOptimizer:
    """
    Optimize inference for batch processing
    """
    
    @staticmethod
    def group_by_expert_allocation(
        inputs: List[Dict[str, Any]],
        routing_predictions: torch.Tensor
    ) -> Dict[str, List[int]]:
        """
        Group inputs by their expert allocation for efficient batch processing
        
        Args:
            inputs: List of input samples
            routing_predictions: Predicted expert routing
            
        Returns:
            Dictionary mapping expert indices to sample indices
        """
        expert_groups = {}
        
        # Get top expert for each sample
        top_experts = torch.argmax(routing_predictions, dim=-1)
        
        for idx, expert_idx in enumerate(top_experts):
            expert_idx = int(expert_idx)
            if expert_idx not in expert_groups:
                expert_groups[expert_idx] = []
            expert_groups[expert_idx].append(idx)
        
        return expert_groups
    
    @staticmethod
    def optimize_expert_loading(
        expert_groups: Dict[str, List[int]],
        memory_limit: int = 8 * 1024 * 1024 * 1024  # 8GB
    ) -> List[List[str]]:
        """
        Optimize expert loading order to minimize memory swapping
        
        Args:
            expert_groups: Mapping of experts to samples
            memory_limit: Available memory in bytes
            
        Returns:
            Ordered list of expert batches to process
        """
        # Sort experts by number of samples (process popular experts first)
        sorted_experts = sorted(
            expert_groups.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        # Group experts that can fit in memory together
        expert_batches = []
        current_batch = []
        current_memory = 0
        expert_memory = memory_limit // 10  # Assume each expert takes ~1/10 of memory
        
        for expert_idx, samples in sorted_experts:
            if current_memory + expert_memory > memory_limit:
                expert_batches.append(current_batch)
                current_batch = [expert_idx]
                current_memory = expert_memory
            else:
                current_batch.append(expert_idx)
                current_memory += expert_memory
        
        if current_batch:
            expert_batches.append(current_batch)
        
        return expert_batches


def create_optimized_model(
    model: nn.Module,
    optimization_config: Dict[str, Any]
) -> nn.Module:
    """
    Apply multiple optimizations to model
    
    Args:
        model: Base model
        optimization_config: Optimization configuration
        
    Returns:
        Optimized model
    """
    # Apply FlashAttention if available
    if optimization_config.get("use_flash_attention", False):
        print("Enabling FlashAttention...")
        # Replace attention modules with FlashAttention
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]
                parent = model
                for part in parent_name.split('.'):
                    if part:
                        parent = getattr(parent, part)
                setattr(parent, attr_name, FlashAttentionWrapper(optimization_config))
    
    # Apply quantization
    if optimization_config.get("quantization", {}).get("enabled", False):
        print("Applying quantization...")
        model = ModelQuantizer.quantize_model(
            model,
            optimization_config["quantization"]
        )
    
    # Apply pruning
    if optimization_config.get("pruning", {}).get("enabled", False):
        print("Applying pruning...")
        model = ModelPruner.structured_pruning(
            model,
            optimization_config["pruning"]
        )
    
    return model 