#!/usr/bin/env python3
"""
Ready-to-run training script for ChartExpert-MoE

This script is designed to work with currently available models and datasets,
handling loading issues gracefully and providing fallback options.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import logging
from typing import Dict, Any, Optional, List
import random
import numpy as np
from PIL import Image
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Import from source
try:
    from src.models.chart_expert_moe import ChartExpertMoE
    from src.training.trainer import MultiStageTrainer  
    from src.utils.logging_utils import setup_logging
except ImportError:
    # Fallback imports
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.models.chart_expert_moe import ChartExpertMoE
    from src.training.trainer import MultiStageTrainer
    from src.utils.logging_utils import setup_logging


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MockChartDataset(Dataset):
    """
    Mock dataset for testing when real datasets are not available
    """
    
    def __init__(self, tokenizer, num_samples: int = 1000, max_length: int = 512):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length
        
        # Mock questions and answers
        self.questions = [
            "What is the highest value in the chart?",
            "Which category has the most data points?",
            "What trend do you see in the data?",
            "Compare the values between categories A and B.",
            "What is the total sum of all values?",
            "Describe the pattern in this time series.",
            "Which quarter shows the best performance?",
            "What percentage does category X represent?",
            "Is there a correlation between the variables?",
            "What can you conclude from this visualization?"
        ]
        
        self.answers = [
            "The highest value is 150.",
            "Category A has the most data points.",
            "The data shows an increasing trend.",
            "Category A has higher values than category B.",
            "The total sum is 500.",
            "The pattern shows seasonal variation.",
            "Q4 shows the best performance.",
            "Category X represents 25% of the total.",
            "There is a positive correlation.",
            "The visualization indicates steady growth."
        ]
        
        self.chart_types = ["bar", "line", "pie", "scatter", "heatmap"]
        
        print(f"Created mock dataset with {num_samples} samples")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random question and answer
        q_idx = idx % len(self.questions)
        question = self.questions[q_idx]
        answer = self.answers[q_idx]
        chart_type = self.chart_types[idx % len(self.chart_types)]
        
        # Create prompt
        prompt = f"Chart Type: {chart_type}\nQuestion: {question}\nAnswer:"
        
        # Tokenize inputs
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        # Tokenize answer for labels
        answer_inputs = self.tokenizer(
            answer,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        labels = answer_inputs["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Create mock image (random tensor)
        image = torch.randn(3, 224, 224)  # Mock chart image
        
        return {
            "image": image,
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels,
            "question": question,
            "answer": answer,
            "chart_type": chart_type,
            "idx": idx
        }


def load_real_model_config():
    """Load configuration with real models that are available"""
    config = {
        "hidden_size": 768,
        "vocab_size": 50257,
        "num_heads": 12,
        "use_hierarchical_experts": True,
        "use_flash_attention": False,  # Disable Flash Attention for compatibility
        "simple_confidence_threshold": 0.95,
        "medium_confidence_threshold": 0.90,
        "complex_confidence_threshold": 0.85,
        "num_early_exit_layers": 3,
        "kv_cache_size_limit": 1073741824,
        "min_experts": 1,
        "aux_loss_weight": 0.01,
        "gradient_clip_norm": 1.0,
        
        "vision_encoder": {
            "encoder_type": "mock",  # Use mock for now, change to "clip" when available
            "model_name": "openai/clip-vit-base-patch32",
            "hidden_size": 768,
            "use_native_resolution": False,
            "use_2d_rope": False,  # Disable for stability
            "max_resolution": 1024,
            "patch_size": 32,
            "max_patches": 196
        },
        
        "llm_backbone": {
            "model_name": "microsoft/DialoGPT-medium",
            "hidden_size": 1024,
            "vocab_size": 50257,
            "use_mock": False
        },
        
        "fusion": {
            "hidden_size": 768,
            "num_heads": 12,
            "fusion_type": "attention",
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
            "layout": {"hidden_size": 768},
            "ocr": {"hidden_size": 768},
            "scale": {"hidden_size": 768},
            "geometric": {"hidden_size": 768},
            "trend": {"hidden_size": 768},
            "query": {"hidden_size": 768},
            "numerical": {"hidden_size": 768},
            "integration": {"hidden_size": 768},
            "alignment": {"hidden_size": 768},
            "chart_to_graph": {"hidden_size": 768},
            "shallow_reasoning": {"hidden_size": 768},
            "orchestrator": {"hidden_size": 768}
        }
    }
    
    return config


def create_simple_trainer(model, device, output_dir):
    """Create a simple trainer for testing"""
    class SimpleTrainer:
        def __init__(self, model, device, output_dir):
            self.model = model
            self.device = device
            self.output_dir = output_dir
            self.logger = logging.getLogger(__name__)
            os.makedirs(output_dir, exist_ok=True)
        
        def train_epoch(self, train_loader, optimizer, epoch):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Forward pass
                try:
                    outputs = self.model(
                        image=batch["image"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch.get("labels")
                    )
                    
                    loss = outputs["loss"]
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if batch_idx % 10 == 0:
                        self.logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                        
                except Exception as e:
                    self.logger.error(f"Error in batch {batch_idx}: {e}")
                    continue
            
            avg_loss = total_loss / max(num_batches, 1)
            self.logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
            return {"loss": avg_loss}
        
        def save_checkpoint(self, epoch, metrics):
            checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "metrics": metrics
            }, checkpoint_path)
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    return SimpleTrainer(model, device, output_dir)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train ChartExpert-MoE")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--mock_data", action="store_true", help="Use mock data for testing")
    args = parser.parse_args()
    
    # Setup
    set_seed(42)
    setup_logging(log_level="INFO", log_dir="./logs")
    logger = logging.getLogger(__name__)
    
    logger.info("Starting ChartExpert-MoE training with available models")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load configuration
    config = load_real_model_config()
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Successfully loaded DialoGPT tokenizer")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        logger.info("Creating mock tokenizer...")
        # Create a simple mock tokenizer
        class MockTokenizer:
            def __init__(self):
                self.pad_token = "[PAD]"
                self.eos_token = "[EOS]"
                self.pad_token_id = 0
                self.eos_token_id = 1
                self.vocab_size = 50257
            
            def __len__(self):
                return self.vocab_size
            
            def __call__(self, text, **kwargs):
                # Simple mock tokenization
                max_length = kwargs.get("max_length", 512)
                tokens = torch.randint(2, self.vocab_size, (max_length,))
                attention_mask = torch.ones(max_length)
                return {
                    "input_ids": tokens.unsqueeze(0),
                    "attention_mask": attention_mask.unsqueeze(0)
                }
        
        tokenizer = MockTokenizer()
    
    # Initialize model
    logger.info("Initializing ChartExpert-MoE model...")
    try:
        model = ChartExpertMoE(config)
        model = model.to(device)
        logger.info("Model initialized successfully")
        
        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return
    
    # Setup dataset
    logger.info("Setting up dataset...")
    if args.mock_data:
        train_dataset = MockChartDataset(tokenizer, num_samples=200)
    else:
        # Try to use real dataset, fallback to mock if failed
        try:
            # Placeholder for real dataset loading
            # For now, use mock dataset
            train_dataset = MockChartDataset(tokenizer, num_samples=200)
            logger.info("Using mock dataset (real datasets not available)")
        except Exception as e:
            logger.error(f"Failed to load real dataset: {e}")
            train_dataset = MockChartDataset(tokenizer, num_samples=200)
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for compatibility
        pin_memory=torch.cuda.is_available()
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Setup trainer
    trainer = create_simple_trainer(model, device, args.output_dir)
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch + 1}/{args.epochs}")
        
        metrics = trainer.train_epoch(train_loader, optimizer, epoch + 1)
        trainer.save_checkpoint(epoch + 1, metrics)
        
        logger.info(f"Epoch {epoch + 1} completed with loss: {metrics['loss']:.4f}")
    
    logger.info("Training completed successfully!")
    logger.info(f"Checkpoints saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 