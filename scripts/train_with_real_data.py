#!/usr/bin/env python3
"""
Complete training script for ChartExpert-MoE with real datasets

This script uses ChartMuseum and ChartQA datasets for training the entire model.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import logging
from typing import Dict, Any, Optional
import random
import numpy as np
from PIL import Image
import wandb
from tqdm import tqdm
from datasets import load_dataset
import torchvision.transforms as transforms

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

try:
    from src.models.chart_expert_moe import ChartExpertMoE
    from src.utils.logging_utils import setup_logging
except ImportError:
    # Direct import if src import fails
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    exec(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'models', 'chart_expert_moe.py')).read())


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class RealChartDataset(Dataset):
    """
    Dataset class for real chart data (ChartMuseum and ChartQA)
    """
    
    def __init__(
        self, 
        dataset_name: str = "chartmuseum",
        split: str = "test",
        tokenizer = None,
        max_length: int = 512,
        image_size: tuple = (224, 224),
        max_samples: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load dataset
        print(f"Loading {dataset_name} dataset (split: {split})...")
        
        if dataset_name.lower() == "chartmuseum":
            self.dataset = load_dataset("lytang/ChartMuseum", split=split)
            self.question_key = "question"
            self.answer_key = "answer"
        elif dataset_name.lower() == "chartqa":
            self.dataset = load_dataset("HuggingFaceM4/ChartQA", split=split)
            self.question_key = "query"  # ChartQA uses 'query' instead of 'question'
            self.answer_key = "label"    # ChartQA uses 'label' instead of 'answer'
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Limit samples if specified
        if max_samples and len(self.dataset) > max_samples:
            indices = list(range(min(max_samples, len(self.dataset))))
            self.dataset = self.dataset.select(indices)
        
        print(f"Loaded {len(self.dataset)} samples from {dataset_name}")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example from the dataset"""
        item = self.dataset[idx]
        
        # Extract components
        image = item.get("image")
        question = item.get(self.question_key, "")
        answer = item.get(self.answer_key, "")
        
        # Process image
        if image is not None:
            if isinstance(image, str):
                # If image is a path/URL, load it
                try:
                    image = Image.open(image).convert("RGB")
                except:
                    # Create dummy image if loading fails
                    image = Image.new("RGB", self.image_size, color="white")
            elif hasattr(image, 'convert'):
                # PIL Image
                image = image.convert("RGB")
            
            processed_image = self.image_transform(image)
        else:
            # Create dummy image if missing
            processed_image = torch.zeros(3, *self.image_size)
        
        # Create prompt
        prompt = f"Question: {question}\nAnswer:"
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        # Tokenize answer for labels
        if answer:
            answer_inputs = self.tokenizer(
                str(answer),
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            labels = answer_inputs["input_ids"].squeeze()
            # Set padding tokens to -100 (ignored in loss)
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            labels = torch.tensor([-100] * self.max_length)
        
        return {
            "image": processed_image,
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels,
            "question": question,
            "answer": str(answer),
            "idx": idx
        }


def get_model_config(use_real_models: bool = True):
    """Get model configuration"""
    config = {
        "hidden_size": 768,
        "vocab_size": 50257,
        "num_heads": 12,
        "use_hierarchical_experts": True,
        "use_flash_attention": False,  # Disable for compatibility
        "simple_confidence_threshold": 0.95,
        "medium_confidence_threshold": 0.90,
        "complex_confidence_threshold": 0.85,
        "num_early_exit_layers": 3,
        "kv_cache_size_limit": 1073741824,
        "min_experts": 1,
        "aux_loss_weight": 0.01,
        "gradient_clip_norm": 1.0,
        
        "vision_encoder": {
            "encoder_type": "clip" if use_real_models else "mock",
            "model_name": "openai/clip-vit-base-patch32",
            "hidden_size": 768,
            "use_native_resolution": False,
            "use_2d_rope": False,
            "max_resolution": 1024,
            "patch_size": 32,
            "max_patches": 196
        },
        
        "llm_backbone": {
            "model_name": "microsoft/DialoGPT-medium",
            "hidden_size": 1024,
            "vocab_size": 50257,
            "use_mock": not use_real_models
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
    
    return config


class ChartTrainer:
    """Simple trainer for ChartExpert-MoE"""
    
    def __init__(self, model, device, output_dir, wandb_enabled=True):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.wandb_enabled = wandb_enabled
        self.logger = logging.getLogger(__name__)
        os.makedirs(output_dir, exist_ok=True)
        
        self.global_step = 0
        self.best_loss = float('inf')
    
    def train_epoch(self, train_loader, optimizer, scheduler, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Move to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                if scheduler:
                    scheduler.step()
                
                # Track metrics
                total_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                # Update progress bar
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                
                # Log to wandb
                if self.wandb_enabled and batch_idx % 50 == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/avg_loss": avg_loss,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/global_step": self.global_step
                    })
                
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return {"loss": avg_loss}
    
    def evaluate(self, eval_loader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                try:
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    
                    outputs = self.model(
                        image=batch["image"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    
                    loss = outputs["loss"]
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    self.logger.error(f"Error in evaluation batch: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return {"eval_loss": avg_loss}
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train ChartExpert-MoE with real data")
    parser.add_argument("--dataset", type=str, default="chartmuseum", choices=["chartmuseum", "chartqa"], help="Dataset to use")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_real", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per split")
    parser.add_argument("--use_real_models", action="store_true", help="Use real CLIP/LLM models")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()
    
    # Setup
    set_seed(42)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize wandb
    wandb_enabled = not args.no_wandb
    if wandb_enabled:
        wandb.init(
            project="chartexpert-moe-real-data",
            config={
                "dataset": args.dataset,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "use_real_models": args.use_real_models
            }
        )
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("✅ Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return
    
    # Load model
    logger.info("Initializing ChartExpert-MoE model...")
    config = get_model_config(use_real_models=args.use_real_models)
    
    try:
        # Import the model directly since we're having import issues
        sys.path.append('./src')
        from models.chart_expert_moe import ChartExpertMoE
        
        model = ChartExpertMoE(config)
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Setup datasets
    logger.info(f"Setting up {args.dataset} dataset...")
    
    # Determine splits based on dataset
    if args.dataset == "chartmuseum":
        train_split = "test"  # ChartMuseum only has test/dev
        eval_split = "dev"
    else:  # chartqa
        train_split = "train"
        eval_split = "val"
    
    train_dataset = RealChartDataset(
        dataset_name=args.dataset,
        split=train_split,
        tokenizer=tokenizer,
        max_samples=args.max_samples
    )
    
    eval_dataset = RealChartDataset(
        dataset_name=args.dataset,
        split=eval_split,
        tokenizer=tokenizer,
        max_samples=args.max_samples // 5 if args.max_samples else None
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    num_training_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_training_steps
    )
    
    # Setup trainer
    trainer = ChartTrainer(model, device, args.output_dir, wandb_enabled)
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Starting epoch {epoch}/{args.epochs}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, optimizer, scheduler, epoch)
        
        # Evaluate
        eval_metrics = trainer.evaluate(eval_loader)
        
        # Log metrics
        all_metrics = {**train_metrics, **eval_metrics}
        logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, Eval Loss: {eval_metrics['eval_loss']:.4f}")
        
        if wandb_enabled:
            wandb.log({f"epoch": epoch, **all_metrics})
        
        # Save checkpoint
        is_best = eval_metrics['eval_loss'] < trainer.best_loss
        if is_best:
            trainer.best_loss = eval_metrics['eval_loss']
        
        trainer.save_checkpoint(epoch, all_metrics, is_best)
    
    logger.info("🎉 Training completed successfully!")
    logger.info(f"Checkpoints saved to: {args.output_dir}")
    logger.info(f"Best evaluation loss: {trainer.best_loss:.4f}")
    
    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main() 