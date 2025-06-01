"""
Data utilities for ChartExpert-MoE
"""

import torch
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for ChartExpert-MoE batches
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Collated batch dictionary
    """
    # Separate different types of data
    images = []
    input_ids = []
    attention_masks = []
    labels = []
    
    for sample in batch:
        if "image" in sample:
            images.append(sample["image"])
        if "input_ids" in sample:
            input_ids.append(sample["input_ids"])
        if "attention_mask" in sample:
            attention_masks.append(sample["attention_mask"])
        if "labels" in sample:
            labels.append(sample["labels"])
    
    # Stack tensors
    collated = {}
    
    if images:
        # Handle both tensor and PIL images
        if isinstance(images[0], torch.Tensor):
            collated["image"] = torch.stack(images)
        else:
            # Convert PIL images to tensors
            from torchvision import transforms
            transform = transforms.ToTensor()
            collated["image"] = torch.stack([transform(img) for img in images])
    
    if input_ids:
        # Pad sequences to same length
        collated["input_ids"] = pad_sequence(input_ids, padding_value=0)
    
    if attention_masks:
        collated["attention_mask"] = pad_sequence(attention_masks, padding_value=0)
    
    if labels:
        collated["labels"] = pad_sequence(labels, padding_value=-100)
    
    return collated


def pad_sequence(
    sequences: List[torch.Tensor],
    padding_value: int = 0,
    max_length: Optional[int] = None
) -> torch.Tensor:
    """
    Pad sequences to the same length
    
    Args:
        sequences: List of tensors to pad
        padding_value: Value to use for padding
        max_length: Maximum length to pad to
        
    Returns:
        Padded tensor
    """
    if not sequences:
        return torch.tensor([])
    
    # Find maximum length
    if max_length is None:
        max_length = max(seq.size(0) for seq in sequences)
    
    # Create padded tensor
    batch_size = len(sequences)
    padded = torch.full((batch_size, max_length), padding_value, dtype=sequences[0].dtype)
    
    # Fill in sequences
    for i, seq in enumerate(sequences):
        length = min(seq.size(0), max_length)
        padded[i, :length] = seq[:length]
    
    return padded


def prepare_batch(
    batch: Dict[str, Any],
    device: torch.device,
    dtype: Optional[torch.dtype] = None
) -> Dict[str, Any]:
    """
    Prepare batch for model input
    
    Args:
        batch: Batch dictionary
        device: Device to move tensors to
        dtype: Optional dtype for floating point tensors
        
    Returns:
        Prepared batch
    """
    prepared = {}
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            prepared[key] = value.to(device)
            
            # Convert dtype for floating point tensors
            if dtype is not None and value.dtype.is_floating_point:
                prepared[key] = prepared[key].to(dtype)
        else:
            prepared[key] = value
    
    return prepared


def create_attention_mask(
    input_ids: torch.Tensor,
    pad_token_id: int = 0
) -> torch.Tensor:
    """
    Create attention mask from input IDs
    
    Args:
        input_ids: Input token IDs
        pad_token_id: ID of padding token
        
    Returns:
        Attention mask (1 for real tokens, 0 for padding)
    """
    return (input_ids != pad_token_id).long()


def preprocess_chart_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> torch.Tensor:
    """
    Preprocess chart image for model input
    
    Args:
        image: PIL Image
        target_size: Target size for resizing
        normalize: Whether to normalize the image
        
    Returns:
        Preprocessed image tensor
    """
    from torchvision import transforms
    
    # Build transform pipeline
    transform_list = [
        transforms.Resize(target_size),
        transforms.ToTensor()
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    transform = transforms.Compose(transform_list)
    return transform(image)


def augment_chart_image(
    image: torch.Tensor,
    augmentation_config: Dict[str, Any]
) -> torch.Tensor:
    """
    Apply data augmentation to chart images
    
    Args:
        image: Image tensor
        augmentation_config: Augmentation configuration
        
    Returns:
        Augmented image tensor
    """
    from torchvision import transforms
    
    augmentations = []
    
    if augmentation_config.get("random_crop", False):
        augmentations.append(transforms.RandomCrop(224))
    
    if augmentation_config.get("color_jitter", False):
        augmentations.append(
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            )
        )
    
    if augmentation_config.get("random_rotation", False):
        augmentations.append(transforms.RandomRotation(degrees=5))
    
    if augmentations:
        augment = transforms.Compose(augmentations)
        image = augment(image)
    
    return image 