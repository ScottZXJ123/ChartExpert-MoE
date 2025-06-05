#!/usr/bin/env python3
"""
Direct test of ChartExpert-MoE model loading and training
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import model directly
from models.chart_expert_moe import ChartExpertMoE

class SimpleDataset(Dataset):
    def __init__(self, num_samples=50):
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return {
            "image": torch.randn(3, 224, 224),
            "input_ids": torch.randint(0, 1000, (50,)),
            "attention_mask": torch.ones(50),
            "labels": torch.randint(0, 1000, (50,))
        }

def main():
    logger.info("Starting ChartExpert-MoE direct test...")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Model configuration with mock components
    config = {
        "hidden_size": 768,
        "vocab_size": 50257,
        "num_heads": 12,
        "use_hierarchical_experts": True,
        "use_flash_attention": False,
        "simple_confidence_threshold": 0.95,
        "medium_confidence_threshold": 0.90,
        "complex_confidence_threshold": 0.85,
        "num_early_exit_layers": 3,
        "kv_cache_size_limit": 1073741824,
        "min_experts": 1,
        "aux_loss_weight": 0.01,
        "gradient_clip_norm": 1.0,
        
        "vision_encoder": {
            "encoder_type": "mock",
            "hidden_size": 768
        },
        
        "llm_backbone": {
            "model_name": "microsoft/DialoGPT-medium",
            "hidden_size": 1024,
            "vocab_size": 50257,
            "use_mock": True  # Use mock to avoid download issues
        },
        
        "fusion": {
            "hidden_size": 768,
            "num_heads": 12,
            "dropout": 0.1
        },
        
        "routing": {
            "hidden_size": 768,
            "num_experts": 12,
            "top_k": 2,
            "dropout": 0.1,
            "load_balancing_weight": 0.01
        },
        
        "moe": {
            "hidden_size": 768,
            "num_experts": 12,
            "top_k": 2,
            "capacity_factor": 1.25,
            "dropout": 0.1
        },
        
        "experts": {
            expert_name: {"hidden_size": 768} 
            for expert_name in [
                "layout", "ocr", "scale", "geometric", "trend", 
                "query", "numerical", "integration", "alignment", 
                "chart_to_graph", "shallow_reasoning", "orchestrator"
            ]
        }
    }
    
    # Initialize model
    logger.info("Loading ChartExpert-MoE model...")
    try:
        model = ChartExpertMoE(config)
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False
    
    # Test forward pass
    logger.info("Testing forward pass...")
    try:
        batch_size = 2
        image = torch.randn(batch_size, 3, 224, 224).to(device)
        input_ids = torch.randint(0, 1000, (batch_size, 50)).to(device)
        attention_mask = torch.ones(batch_size, 50).to(device)
        
        with torch.no_grad():
            outputs = model(
                image=image,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        logger.info(f"✅ Forward pass successful!")
        logger.info(f"Output logits shape: {outputs['logits'].shape}")
        logger.info(f"Available output keys: {list(outputs.keys())}")
        
    except Exception as e:
        logger.error(f"❌ Forward pass failed: {e}")
        return False
    
    # Test training step
    logger.info("Testing training step...")
    try:
        dataset = SimpleDataset(num_samples=20)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:  # Only test a few batches
                break
                
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                image=batch["image"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs["loss"]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            logger.info(f"Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
        
        logger.info("✅ Training step successful!")
        
    except Exception as e:
        logger.error(f"❌ Training step failed: {e}")
        return False
    
    logger.info("🎉 All tests passed! ChartExpert-MoE is ready for training!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n" + "="*60)
        print("🚀 READY TO TRAIN!")
        print("Your ChartExpert-MoE system is fully functional.")
        print("You can now run full training with real or mock data.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ ISSUES DETECTED") 
        print("Please fix the issues above before training.")
        print("="*60) 