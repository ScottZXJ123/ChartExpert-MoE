"""
Model utilities for ChartExpert-MoE
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_size(model: nn.Module, unit: str = "MB") -> float:
    """
    Get the size of a model in memory
    
    Args:
        model: PyTorch model
        unit: Unit for size (B, KB, MB, GB)
        
    Returns:
        Model size in specified unit
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_bytes = param_size + buffer_size
    
    units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 ** 2,
        "GB": 1024 ** 3
    }
    
    if unit not in units:
        raise ValueError(f"Unknown unit: {unit}")
    
    return size_bytes / units[unit]


def get_parameter_stats(model: nn.Module) -> Dict[str, Any]:
    """
    Get detailed parameter statistics for a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter statistics
    """
    stats = {
        "total_params": count_parameters(model, trainable_only=False),
        "trainable_params": count_parameters(model, trainable_only=True),
        "model_size_mb": get_model_size(model, unit="MB"),
        "parameter_groups": {}
    }
    
    # Count parameters by module type
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type not in stats["parameter_groups"]:
            stats["parameter_groups"][module_type] = 0
        
        for param in module.parameters(recurse=False):
            stats["parameter_groups"][module_type] += param.numel()
    
    # Expert-specific statistics
    expert_params = {}
    for name, param in model.named_parameters():
        if "expert" in name:
            expert_name = name.split(".")[1] if "." in name else "unknown"
            if expert_name not in expert_params:
                expert_params[expert_name] = 0
            expert_params[expert_name] += param.numel()
    
    stats["expert_params"] = expert_params
    
    return stats


def freeze_module(module: nn.Module):
    """
    Freeze all parameters in a module
    
    Args:
        module: Module to freeze
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module):
    """
    Unfreeze all parameters in a module
    
    Args:
        module: Module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True


def get_activation_memory(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32
) -> float:
    """
    Estimate activation memory for a given input shape
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        dtype: Data type
        
    Returns:
        Estimated activation memory in MB
    """
    # This is a simplified estimation
    # In practice, you'd need to trace through the model
    
    # Rough estimate: 2-3x model size for activations
    model_size_mb = get_model_size(model, unit="MB")
    batch_size = input_shape[0] if len(input_shape) > 0 else 1
    
    # Scale by batch size and add overhead
    activation_memory_mb = model_size_mb * 2.5 * batch_size
    
    return activation_memory_mb 