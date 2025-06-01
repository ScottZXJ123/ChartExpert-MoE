"""
Data preprocessing utilities for ChartExpert-MoE
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class ChartPreprocessor:
    """Main preprocessor for chart data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_processor = ImageProcessor(config)
        self.text_processor = TextProcessor(config)
        
    def preprocess(
        self,
        image: Image.Image,
        text: str,
        augment: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Preprocess chart image and text"""
        # Process image
        image_tensor = self.image_processor.process(image, augment=augment)
        
        # Process text
        text_features = self.text_processor.process(text)
        
        return {
            "image": image_tensor,
            "input_ids": text_features["input_ids"],
            "attention_mask": text_features["attention_mask"]
        }


class ImageProcessor:
    """Image preprocessing for charts"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_size = config.get("image_size", [224, 224])
        self.use_native_resolution = config.get("use_native_resolution", False)
        
        # Basic transforms
        self.basic_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Augmentation transforms
        augmentation_config = config.get("augmentation", {})
        self.augment_transform = transforms.Compose([
            transforms.RandomRotation(augmentation_config.get("rotation_range", 5)),
            transforms.ColorJitter(
                brightness=augmentation_config.get("brightness_range", 0.1),
                contrast=augmentation_config.get("contrast_range", 0.1),
                saturation=augmentation_config.get("color_jitter", 0.05),
                hue=0.02
            ),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def process(self, image: Image.Image, augment: bool = False) -> torch.Tensor:
        """Process a single image"""
        if augment and self.config.get("augmentation", {}).get("enabled", False):
            return self.augment_transform(image)
        else:
            return self.basic_transform(image)
            
    def add_chart_specific_augmentation(
        self,
        image: Image.Image
    ) -> Image.Image:
        """Apply chart-specific augmentations"""
        # This is a placeholder for chart-specific augmentations
        # like axis noise, legend position variation, etc.
        return image


class TextProcessor:
    """Text preprocessing for chart queries"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_length = config.get("max_length", 512)
        
        # Mock tokenizer (in practice, would use actual tokenizer)
        self.vocab_size = config.get("vocab_size", 32000)
        
    def process(self, text: str) -> Dict[str, torch.Tensor]:
        """Process text input"""
        # Mock tokenization (in practice, would use actual tokenizer)
        tokens = text.lower().split()[:self.max_length]
        
        # Convert to token ids (mock)
        input_ids = torch.randint(0, self.vocab_size, (len(tokens),))
        
        # Pad to max length
        if len(input_ids) < self.max_length:
            padding = torch.zeros(self.max_length - len(input_ids), dtype=torch.long)
            input_ids = torch.cat([input_ids, padding])
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        attention_mask[len(tokens):] = 0
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token ids to text"""
        # Mock decoding
        return " ".join([f"token_{id}" for id in token_ids if id != 0]) 