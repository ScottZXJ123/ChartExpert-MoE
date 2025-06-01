"""
Optimizer and scheduler utilities for ChartExpert-MoE training
"""

import torch
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    LinearLR, 
    OneCycleLR,
    _LRScheduler
)
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from typing import Optional, List, Dict, Any


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    optimizer_type: str = "adamw",
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8
) -> torch.optim.Optimizer:
    """
    Create optimizer with parameter-specific settings
    
    Args:
        model: Model to optimize
        learning_rate: Base learning rate
        optimizer_type: Type of optimizer (adamw, adam, sgd)
        weight_decay: Weight decay coefficient
        betas: Adam beta parameters
        eps: Adam epsilon
        
    Returns:
        Configured optimizer
    """
    # Separate parameters that should and shouldn't have weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Don't apply weight decay to bias, LayerNorm, or embedding parameters
        if any(nd in name for nd in ["bias", "LayerNorm", "layernorm", "embed"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]
    
    # Create optimizer
    if optimizer_type.lower() == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps
        )
    elif optimizer_type.lower() == "adam":
        optimizer = Adam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = SGD(
            optimizer_grouped_parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    min_lr: float = 1e-7,
    **kwargs
) -> _LRScheduler:
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler (cosine, linear, onecycle)
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        min_lr: Minimum learning rate
        
    Returns:
        Configured scheduler
    """
    if scheduler_type.lower() == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type.lower() == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type.lower() == "cosine_annealing":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - num_warmup_steps,
            eta_min=min_lr
        )
        # Add warmup if needed
        if num_warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=num_warmup_steps
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[num_warmup_steps]
            )
    elif scheduler_type.lower() == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]["lr"],
            total_steps=num_training_steps,
            pct_start=num_warmup_steps / num_training_steps if num_warmup_steps > 0 else 0.3,
            anneal_strategy="cos",
            final_div_factor=1000
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


class SequentialLR(_LRScheduler):
    """Sequential learning rate scheduler (for warmup + main scheduler)"""
    
    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1):
        self.schedulers = schedulers
        self.milestones = milestones
        self.last_epoch = last_epoch
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        idx = 0
        for i, milestone in enumerate(self.milestones):
            if self.last_epoch >= milestone:
                idx = i + 1
        
        if idx >= len(self.schedulers):
            idx = len(self.schedulers) - 1
            
        return self.schedulers[idx].get_lr()
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        idx = 0
        for i, milestone in enumerate(self.milestones):
            if epoch >= milestone:
                idx = i + 1
        
        if idx >= len(self.schedulers):
            idx = len(self.schedulers) - 1
            
        self.schedulers[idx].step(epoch)


def get_parameter_groups(
    model: torch.nn.Module,
    learning_rate: float,
    backbone_lr_multiplier: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with different learning rates
    
    Args:
        model: Model to create groups for
        learning_rate: Base learning rate
        backbone_lr_multiplier: Multiplier for backbone parameters
        
    Returns:
        List of parameter groups
    """
    backbone_params = []
    expert_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if "vision_encoder" in name or "llm_backbone" in name:
            backbone_params.append(param)
        elif "expert" in name:
            expert_params.append(param)
        else:
            other_params.append(param)
    
    parameter_groups = [
        {
            "params": backbone_params,
            "lr": learning_rate * backbone_lr_multiplier,
            "name": "backbone"
        },
        {
            "params": expert_params,
            "lr": learning_rate,
            "name": "experts"
        },
        {
            "params": other_params,
            "lr": learning_rate,
            "name": "other"
        }
    ]
    
    # Remove empty groups
    parameter_groups = [g for g in parameter_groups if len(g["params"]) > 0]
    
    return parameter_groups 