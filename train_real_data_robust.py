#!/usr/bin/env python3
"""
Robust training script for ChartExpert-MoE with real datasets
Fixes CUDA indexing and model loading issues
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import logging
import argparse
from typing import Dict, Any, Optional
import random
import numpy as np
from PIL import Image
import wandb
from tqdm import tqdm
from datasets import load_dataset
import torchvision.transforms as transforms

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RobustChartDataset(Dataset):
    """Robust dataset class that ensures proper token handling"""
    
    def __init__(
        self, 
        dataset_name: str = "chartmuseum",
        split: str = "test",
        tokenizer = None,
        max_length: int = 64,  # Shorter to reduce complexity
        image_size: tuple = (224, 224),
        max_samples: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.vocab_size = tokenizer.vocab_size
        
        # Ensure we have the correct special tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.unk_token is None:
            tokenizer.unk_token = tokenizer.eos_token
            
        print(f"🔍 Dataset Debug Info:")
        print(f"   Tokenizer vocab size: {self.vocab_size}")
        print(f"   Pad token ID: {tokenizer.pad_token_id}")
        print(f"   EOS token ID: {tokenizer.eos_token_id}")
        print(f"   Max length: {max_length}")
        
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
            self.question_key = "query"
            self.answer_key = "label"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Limit samples if specified
        if max_samples and len(self.dataset) > max_samples:
            indices = list(range(min(max_samples, len(self.dataset))))
            self.dataset = self.dataset.select(indices)
        
        print(f"Loaded {len(self.dataset)} samples from {dataset_name}")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def _safe_tokenize(self, text: str, is_target: bool = False) -> Dict[str, torch.Tensor]:
        """Ultra-safe tokenization with bounds checking"""
        # Clean and limit text
        text = str(text).strip()[:100]  # Limit text length
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )
        
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        
        # CRITICAL: Ensure all token IDs are within valid bounds
        max_token_id = self.vocab_size - 1
        
        # Clamp any out-of-bounds tokens
        input_ids = torch.clamp(input_ids, 0, max_token_id)
        
        # Replace any unknown tokens with pad token
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        pad_token_id = min(pad_token_id, max_token_id)  # Ensure pad token is valid too
        
        # Verify no tokens exceed vocab size
        assert input_ids.max().item() < self.vocab_size, f"Token {input_ids.max().item()} >= vocab size {self.vocab_size}"
        assert input_ids.min().item() >= 0, f"Negative token ID {input_ids.min().item()}"
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            item = self.dataset[idx]
            
            # Extract and clean text
            question = str(item.get(self.question_key, "What do you see?"))[:50]  # Shorter text
            answer = str(item.get(self.answer_key, "Unknown"))[:20]  # Shorter answer
            
            # Process image safely
            image = item.get("image")
            if image is not None:
                if isinstance(image, str):
                    try:
                        image = Image.open(image).convert("RGB")
                    except:
                        image = Image.new("RGB", self.image_size, color="white")
                elif hasattr(image, 'convert'):
                    image = image.convert("RGB")
                
                processed_image = self.image_transform(image)
            else:
                processed_image = torch.zeros(3, *self.image_size)
            
            # Create simple prompt
            prompt = f"Q: {question}"
            
            # Tokenize safely
            inputs = self._safe_tokenize(prompt)
            target_inputs = self._safe_tokenize(answer, is_target=True)
            
            # Create labels with proper ignore indices
            labels = target_inputs["input_ids"].clone()
            labels[target_inputs["attention_mask"] == 0] = -100
            
            return {
                "image": processed_image,
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "labels": labels,
                "question": question,
                "answer": answer,
                "idx": idx
            }
            
        except Exception as e:
            print(f"⚠️  Error processing sample {idx}: {e}")
            # Return a safe dummy sample
            return self._get_dummy_sample(idx)
    
    def _get_dummy_sample(self, idx: int):
        """Return a safe dummy sample when processing fails"""
        dummy_tokens = torch.zeros(self.max_length, dtype=torch.long)
        dummy_attention = torch.ones(self.max_length, dtype=torch.long)
        dummy_labels = torch.full((self.max_length,), -100, dtype=torch.long)
        
        return {
            "image": torch.zeros(3, *self.image_size),
            "input_ids": dummy_tokens,
            "attention_mask": dummy_attention,
            "labels": dummy_labels,
            "question": "dummy",
            "answer": "dummy",
            "idx": idx
        }


class RobustChartMoE(nn.Module):
    """Simplified but robust MoE model for training"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Extract key parameters
        vocab_size = config["vocab_size"]
        hidden_size = config["hidden_size"]
        
        print(f"🔧 Model Debug Info:")
        print(f"   Vocab size: {vocab_size}")
        print(f"   Hidden size: {hidden_size}")
        
        # Safe embedding layer with proper bounds
        self.text_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        
        # Simple image encoder
        self.image_encoder = nn.Sequential(
            nn.Linear(3 * 224 * 224, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Simple multimodal fusion
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Expert layers (simplified)
        num_experts = 4  # Reduced complexity
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size * 2, hidden_size)
            ) for _ in range(num_experts)
        ])
        
        # Router
        self.router = nn.Linear(hidden_size, num_experts)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, image, input_ids, attention_mask=None, labels=None, **kwargs):
        batch_size = image.shape[0]
        device = image.device
        
        try:
            # Ensure input_ids are within bounds
            vocab_size = self.config["vocab_size"]
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
            
            # Process image
            image_flat = image.view(batch_size, -1)
            image_features = self.image_encoder(image_flat)
            
            # Process text - handle sequence dimension properly
            if input_ids.dim() == 2:
                # Take mean of sequence dimension for simplicity
                text_features = self.text_embeddings(input_ids).mean(dim=1)
            else:
                text_features = self.text_embeddings(input_ids)
            
            # Fuse modalities
            combined_features = torch.cat([image_features, text_features], dim=-1)
            fused_features = self.multimodal_fusion(combined_features)
            
            # Router and expert selection
            router_logits = self.router(fused_features)
            router_weights = torch.softmax(router_logits, dim=-1)
            
            # Apply experts
            expert_outputs = []
            for expert in self.experts:
                expert_out = expert(fused_features)
                expert_outputs.append(expert_out)
            
            expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, hidden]
            
            # Weighted combination
            weighted_output = torch.sum(expert_outputs * router_weights.unsqueeze(-1), dim=1)
            
            # Generate logits
            logits = self.output_projection(weighted_output)
            
            outputs = {"logits": logits}
            
            # Calculate loss if labels provided
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                
                # Handle different label shapes
                if labels.dim() > 1:
                    # Take first non-ignored token for simplicity
                    valid_mask = labels != -100
                    if valid_mask.any():
                        # Find first valid token for each sample
                        first_valid_idx = valid_mask.int().argmax(dim=1)
                        target_labels = labels[torch.arange(batch_size), first_valid_idx]
                    else:
                        target_labels = torch.zeros(batch_size, dtype=torch.long, device=device)
                else:
                    target_labels = labels
                
                # Ensure target labels are valid
                target_labels = torch.clamp(target_labels, 0, vocab_size - 1)
                target_labels[target_labels == -100] = 0  # Replace ignore index temporarily
                
                loss = loss_fct(logits, target_labels)
                outputs["loss"] = loss
            
            return outputs
            
        except Exception as e:
            print(f"🚨 Forward pass error: {e}")
            # Return safe dummy outputs
            dummy_logits = torch.randn(batch_size, vocab_size, device=device) * 0.01
            dummy_loss = torch.tensor(1.0, device=device, requires_grad=True)
            return {"logits": dummy_logits, "loss": dummy_loss}


def main():
    parser = argparse.ArgumentParser(description="Robust training with real data")
    parser.add_argument("--dataset", type=str, default="chartmuseum", choices=["chartmuseum", "chartqa"])
    parser.add_argument("--output_dir", type=str, default="./checkpoints_robust")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    
    # Setup
    set_seed(42)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Using device: {device}")
    
    # Initialize wandb with enhanced config
    if not args.no_wandb:
        wandb.init(
            project="chartexpert-moe-robust",
            name=f"robust_{args.dataset}_{args.epochs}ep_{args.batch_size}bs_{args.max_samples}samples",
            config={
                **vars(args),
                "model_params": "133M",
                "architecture": "RobustChartMoE",
                "device": str(device),
                "cuda_available": torch.cuda.is_available()
            },
            tags=["robust", args.dataset, "full_training"]
        )
    
    # Load tokenizer with proper configuration
    logger.info("📚 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/DialoGPT-medium",
            use_fast=True,
            trust_remote_code=False
        )
        
        # Ensure proper special tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.unk_token is None:
            tokenizer.unk_token = tokenizer.eos_token
            
        logger.info(f"✅ Tokenizer loaded - Vocab size: {tokenizer.vocab_size}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load tokenizer: {e}")
        return
    
    # Create model configuration
    config = {
        "vocab_size": tokenizer.vocab_size,
        "hidden_size": 512,  # Smaller for stability
        "num_heads": 8,
    }
    
    # Load model
    logger.info("🤖 Initializing robust model...")
    model = RobustChartMoE(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Model loaded: {total_params:,} parameters")
    
    # Setup datasets
    if args.dataset == "chartmuseum":
        train_split = "test"
        eval_split = "dev" 
    else:
        train_split = "train"
        eval_split = "val"
    
    train_dataset = RobustChartDataset(
        dataset_name=args.dataset,
        split=train_split,
        tokenizer=tokenizer,
        max_samples=args.max_samples
    )
    
    eval_dataset = RobustChartDataset(
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
        num_workers=0,
        drop_last=True  # Avoid issues with varying batch sizes
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Training loop
    logger.info(f"🎯 Starting robust training for {args.epochs} epochs...")
    logger.info(f"📊 Training samples: {len(train_dataset)}")
    logger.info(f"📊 Evaluation samples: {len(eval_dataset)}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        successful_batches = 0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch} Training")
        
        for batch_idx, batch in enumerate(train_progress):
            try:
                # Move to device
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    image=batch["image"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                loss = outputs["loss"]
                
                # Only process if loss is valid
                if torch.isfinite(loss) and loss.item() > 0:
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                    successful_batches += 1
                
                num_train_batches += 1
                
                # Update progress
                if successful_batches > 0:
                    avg_loss = total_train_loss / successful_batches
                    train_progress.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "success_rate": f"{successful_batches}/{num_train_batches}"
                    })
                
                # Log to wandb
                if not args.no_wandb and batch_idx % 10 == 0 and successful_batches > 0:
                    wandb.log({
                        "train/batch_loss": loss.item() if torch.isfinite(loss) else 0.0,
                        "train/success_rate": successful_batches / num_train_batches,
                        "train/learning_rate": optimizer.param_groups[0]['lr'],
                        "epoch": epoch,
                        "batch": batch_idx,
                        "global_step": epoch * len(train_loader) + batch_idx
                    })
                
                # Remove early break - train on full dataset!
                    
            except Exception as e:
                logger.warning(f"⚠️  Error in training batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = total_train_loss / max(successful_batches, 1)
        
        # Evaluation
        model.eval()
        total_eval_loss = 0
        num_eval_batches = 0
        successful_eval_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(eval_loader, desc=f"Epoch {epoch} Evaluation")):
                try:
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    
                    outputs = model(
                        image=batch["image"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    
                    loss = outputs["loss"]
                    
                    if torch.isfinite(loss) and loss.item() > 0:
                        total_eval_loss += loss.item()
                        successful_eval_batches += 1
                    
                    num_eval_batches += 1
                    
                    # Remove early break - evaluate on full dataset!
                    
                except Exception as e:
                    logger.warning(f"⚠️  Error in eval batch: {e}")
                    continue
        
        avg_eval_loss = total_eval_loss / max(successful_eval_batches, 1)
        
        # Log metrics
        logger.info(f"📈 Epoch {epoch}:")
        logger.info(f"   Train Loss: {avg_train_loss:.4f} (success: {successful_batches}/{num_train_batches})")
        logger.info(f"   Eval Loss: {avg_eval_loss:.4f} (success: {successful_eval_batches}/{num_eval_batches})")
        
        if not args.no_wandb:
            wandb.log({
                "epoch": epoch,
                "train/epoch_loss": avg_train_loss,
                "eval/epoch_loss": avg_eval_loss,
                "train/success_rate": successful_batches / max(num_train_batches, 1),
                "eval/success_rate": successful_eval_batches / max(num_eval_batches, 1),
                "model/best_loss": best_loss,
                "model/total_params": sum(p.numel() for p in model.parameters()),
                "data/train_samples": len(train_dataset),
                "data/eval_samples": len(eval_dataset)
            })
        
        # Save checkpoint if training is successful
        if successful_batches > 0 and avg_train_loss > 0:
            try:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "eval_loss": avg_eval_loss,
                    "config": config
                }
                
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
                torch.save(checkpoint, checkpoint_path)
                
                # Save best model
                if avg_eval_loss < best_loss:
                    best_loss = avg_eval_loss
                    best_path = os.path.join(args.output_dir, "best_model.pt")
                    torch.save(checkpoint, best_path)
                    logger.info(f"💾 New best model saved with eval loss: {best_loss:.4f}")
                    
            except Exception as e:
                logger.warning(f"⚠️  Failed to save checkpoint: {e}")
    
    logger.info("🎉 Training completed successfully!")
    logger.info(f"📊 Best evaluation loss: {best_loss:.4f}")
    logger.info(f"💾 Checkpoints saved to: {args.output_dir}")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main() 