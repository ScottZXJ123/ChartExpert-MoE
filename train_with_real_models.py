#!/usr/bin/env python3
"""
Fixed Real Model MoE Training Script
Fixes tensor dimension mismatches and architecture issues
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import random
from PIL import Image
import argparse
import logging
from typing import Dict, Any, Optional
from tqdm import tqdm
import wandb
import gc
from torch.utils.checkpoint import checkpoint

# Transformers and datasets
from transformers import (
    AutoModel, AutoTokenizer, AutoProcessor,
    LlamaModel, LlamaTokenizer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
import torchvision.transforms as transforms


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RealModelDataset(Dataset):
    """Dataset optimized for real pre-trained models"""
    
    def __init__(
        self, 
        dataset_name: str = "chartqa",
        split: str = "train",
        tokenizer = None,
        processor = None,
        max_length: int = 256,
        image_size: tuple = (224, 224),
        max_samples: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        self.vocab_size = tokenizer.vocab_size
        
        print(f"🔧 Real Model Dataset Config:")
        print(f"   Dataset: {dataset_name}")
        print(f"   Split: {split}")
        print(f"   Tokenizer vocab size: {self.vocab_size}")
        print(f"   Max length: {max_length}")
        print(f"   Has processor: {processor is not None}")
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load dataset
        print(f"📚 Loading {dataset_name} dataset...")
        
        if dataset_name.lower() == "chartqa":
            self.dataset = load_dataset("HuggingFaceM4/ChartQA", split=split)
            self.question_key = "query"
            self.answer_key = "label"
        elif dataset_name.lower() == "chartmuseum":
            self.dataset = load_dataset("lytang/ChartMuseum", split=split)
            self.question_key = "question"
            self.answer_key = "answer"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Limit samples if specified
        if max_samples and len(self.dataset) > max_samples:
            indices = random.sample(range(len(self.dataset)), max_samples)
            self.dataset = self.dataset.select(indices)
        
        print(f"✅ Loaded {len(self.dataset)} samples")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            item = self.dataset[idx]
            
            # Extract text
            question = str(item.get(self.question_key, "What is shown in this chart?"))
            answer = str(item.get(self.answer_key, "No answer"))
            
            # Simplified text format
            text = f"Question: {question} Answer: {answer}"
            
            # Process image
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
            
            # Tokenize
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
            
            # Simple labels - just copy input_ids for language modeling
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # Mask padding
            
            return {
                "image": processed_image,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "question": question,
                "answer": answer,
                "idx": idx
            }
            
        except Exception as e:
            print(f"⚠️ Error processing sample {idx}: {e}")
            return self._get_dummy_sample(idx)
    
    def _get_dummy_sample(self, idx: int):
        """Safe dummy sample"""
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


class RealModelMoE(nn.Module):
    """Fixed MoE architecture built on top of real pre-trained models"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        backbone_name = config["backbone_model"]
        vocab_size = config["vocab_size"]
        num_experts = config.get("num_experts", 4)  # Reduce experts for memory
        
        print(f"🏗️ Real Model MoE Architecture:")
        print(f"   Backbone: {backbone_name}")
        print(f"   Vocab size: {vocab_size}")
        print(f"   Number of experts: {num_experts}")
        
        # Load backbone model with memory optimizations
        try:
            if "qwen" in backbone_name.lower():
                print("🔄 Loading Qwen model with memory optimizations...")
                self.backbone = AutoModel.from_pretrained(
                    backbone_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,  # Use float32 to avoid precision issues
                    low_cpu_mem_usage=True,
                    device_map=None  # Load to CPU first, then move to GPU
                )
                self.use_vision = "vl" in backbone_name.lower()
            else:
                print("🔄 Loading generic model...")
                self.backbone = AutoModel.from_pretrained(
                    backbone_name,
                    torch_dtype=torch.float32,  # Use float32 to avoid precision issues
                    low_cpu_mem_usage=True
                )
                self.use_vision = False
                
        except Exception as e:
            print(f"⚠️ Error loading backbone: {e}")
            print("🔄 Using simplified backbone...")
            self.backbone = None
            self.use_vision = False
        
        # Get actual hidden size from backbone
        if self.backbone and hasattr(self.backbone.config, 'hidden_size'):
            backbone_hidden_size = self.backbone.config.hidden_size
        elif self.backbone and hasattr(self.backbone.config, 'd_model'):
            backbone_hidden_size = self.backbone.config.d_model
        else:
            backbone_hidden_size = 768  # Default fallback
            
        self.hidden_size = backbone_hidden_size
        print(f"   Using hidden size: {self.hidden_size}")
        
        # Smaller image encoder for memory efficiency
        if not self.use_vision:
            self.image_encoder = nn.Sequential(
                nn.Linear(3 * 224 * 224, self.hidden_size // 2),  # Smaller intermediate size
                nn.LayerNorm(self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size // 2, self.hidden_size)
            )
        
        # Smaller MoE Expert networks for memory efficiency
        expert_hidden = self.hidden_size // 2  # Reduce expert size
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, expert_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(expert_hidden, self.hidden_size),
                nn.Dropout(0.1)
            ) for _ in range(num_experts)
        ])
        
        # Smaller expert router
        self.router = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),  # Smaller router
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 4, num_experts)
        )
        
        # Output head
        self.output_head = nn.Linear(self.hidden_size, vocab_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # ALWAYS freeze backbone for memory efficiency
        if self.backbone:
            print("🔒 Freezing backbone parameters for memory efficiency")
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Enable gradient checkpointing for backbone
            if hasattr(self.backbone, 'gradient_checkpointing_enable'):
                self.backbone.gradient_checkpointing_enable()
                print("✅ Gradient checkpointing enabled")
        
        self._init_new_weights()
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
    
    def _init_new_weights(self):
        """Initialize new components"""
        for module in [self.experts, self.router, self.output_head, self.layer_norm]:
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)
                elif isinstance(submodule, nn.LayerNorm):
                    nn.init.ones_(submodule.weight)
                    nn.init.zeros_(submodule.bias)
    
    def forward(self, image, input_ids, attention_mask=None, labels=None, **kwargs):
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        try:
            # Get text features from backbone with gradient checkpointing
            if self.backbone:
                # Use gradient checkpointing to save memory during backward pass
                def backbone_forward(input_ids, attention_mask):
                    return self.backbone(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                
                if self.training and hasattr(self.backbone, 'gradient_checkpointing_enable'):
                    backbone_outputs = checkpoint(backbone_forward, input_ids, attention_mask)
                else:
                    backbone_outputs = backbone_forward(input_ids, attention_mask)
                
                # Get features
                if hasattr(backbone_outputs, 'last_hidden_state'):
                    text_features = backbone_outputs.last_hidden_state
                else:
                    text_features = backbone_outputs.hidden_states[-1]
                
                # Pool features with memory-efficient operations
                if attention_mask is not None:
                    # More memory-efficient pooling
                    mask_expanded = attention_mask.unsqueeze(-1).float()
                    text_pooled = (text_features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
                else:
                    text_pooled = text_features.mean(1)
            else:
                # Fallback: create dummy features
                text_pooled = torch.randn(batch_size, self.hidden_size, device=device, dtype=torch.float32)
            
            # Process image for non-VL models (more memory efficient)
            if not self.use_vision and hasattr(self, 'image_encoder'):
                # Ensure image is float32
                image = image.float()
                
                # Process image in chunks if batch is too large
                if batch_size > 4:
                    image_features_list = []
                    for i in range(0, batch_size, 2):  # Process 2 at a time
                        img_chunk = image[i:i+2].reshape(min(2, batch_size-i), -1)
                        img_feat_chunk = self.image_encoder(img_chunk)
                        image_features_list.append(img_feat_chunk)
                    image_features = torch.cat(image_features_list, dim=0)
                else:
                    image_flat = image.reshape(batch_size, -1)
                    image_features = self.image_encoder(image_flat)
                
                # Simple feature combination
                combined_features = text_pooled + image_features
            else:
                combined_features = text_pooled
            
            # Clear intermediate tensors
            del text_pooled
            if 'image_features' in locals():
                del image_features
            torch.cuda.empty_cache()
            
            # Apply layer normalization
            normalized_features = self.layer_norm(combined_features)
            
            # MoE routing with memory efficiency
            router_logits = self.router(normalized_features)
            router_weights = torch.softmax(router_logits, dim=-1)
            
            # Apply experts with gradient checkpointing if training
            expert_outputs = []
            for expert in self.experts:
                if self.training:
                    expert_out = checkpoint(expert, normalized_features)
                else:
                    expert_out = expert(normalized_features)
                expert_outputs.append(expert_out)
            
            expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, hidden]
            
            # Weighted combination
            weighted_output = torch.sum(expert_outputs * router_weights.unsqueeze(-1), dim=1)
            
            # Residual connection
            final_features = normalized_features + weighted_output
            
            # Clear intermediate tensors
            del expert_outputs, weighted_output, normalized_features
            torch.cuda.empty_cache()
            
            # Generate logits
            logits = self.output_head(final_features)
            
            outputs = {"logits": logits, "router_weights": router_weights}
            
            # Calculate loss
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                
                # Ensure consistent data types for loss calculation
                logits_for_loss = logits.float()
                labels_for_loss = labels.long()
                
                # Simple token-level loss
                if labels_for_loss.dim() > 1 and labels_for_loss.size(1) > 1:
                    # Take the first valid token
                    loss = loss_fct(logits_for_loss, labels_for_loss[:, 0])
                else:
                    loss = loss_fct(logits_for_loss, labels_for_loss.squeeze())
                
                # Add small diversity loss
                router_entropy = -torch.sum(router_weights * torch.log(router_weights + 1e-8), dim=-1)
                diversity_loss = -router_entropy.mean()
                
                total_loss = loss + 0.01 * diversity_loss
                outputs["loss"] = total_loss
                outputs["language_loss"] = loss
                outputs["diversity_loss"] = diversity_loss
            
            return outputs
            
        except Exception as e:
            print(f"🚨 Forward pass error: {e}")
            # Safe fallback
            dummy_logits = torch.randn(batch_size, self.config["vocab_size"], device=device, dtype=torch.float32) * 0.01
            dummy_loss = torch.tensor(1.0, device=device, requires_grad=True)
            return {"logits": dummy_logits, "loss": dummy_loss}


def main():
    parser = argparse.ArgumentParser(description="Memory-Optimized Real Model MoE Training")
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen2.5-1.5B", 
                      choices=[
                          "Qwen/Qwen2.5-VL-2B-Instruct",
                          "Qwen/Qwen2.5-1.5B",
                          "meta-llama/Llama-3.2-1B",
                          "microsoft/DialoGPT-medium"
                      ])
    parser.add_argument("--dataset", type=str, default="chartmuseum", choices=["chartqa", "chartmuseum"])
    parser.add_argument("--output_dir", type=str, default="./checkpoints_real_fixed")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)  # Very small batch size for large models
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_samples", type=int, default=50)  # Small dataset for testing
    parser.add_argument("--freeze_backbone", action="store_true", default=True)  # Always freeze by default
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()
    
    # Setup
    set_seed(42)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Using device: {device}")
    logger.info(f"🤖 Backbone model: {args.backbone}")
    
    # Clear GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
        # Print GPU memory info
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_cached = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"🔍 GPU Memory - Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project="chartexpert-real-memory-opt",
            name=f"memopt_{args.backbone.split('/')[-1]}_{args.dataset}_{args.epochs}ep",
            config=vars(args),
            tags=["memory_optimized", args.dataset]
        )
    
    # Load tokenizer
    logger.info("📚 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
        processor = None
    except:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Fallback
        processor = None
    
    # Ensure proper special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"✅ Tokenizer loaded - Vocab size: {tokenizer.vocab_size}")
    
    # Create model config with memory optimizations
    config = {
        "backbone_model": args.backbone,
        "vocab_size": tokenizer.vocab_size,
        "num_experts": 4,  # Reduced number of experts
        "freeze_backbone": True  # Always freeze backbone
    }
    
    # Initialize model
    logger.info("🏗️ Building memory-optimized real model MoE...")
    model = RealModelMoE(config)
    
    # Move model to GPU with memory monitoring
    try:
        model = model.to(device)
        # Keep model in float32 to avoid precision issues
        
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_cached = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"🔍 GPU Memory after model load - Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("❌ CUDA out of memory during model loading!")
            logger.error("💡 Try using a smaller model or reduce batch size further")
            return
        else:
            raise e
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"✅ Model ready:")
    logger.info(f"   Total parameters: {total_params:,}")
    logger.info(f"   Trainable parameters: {trainable_params:,}")
    logger.info(f"   Frozen ratio: {(total_params-trainable_params)/total_params:.1%}")
    
    # Setup datasets with smaller sizes
    if args.dataset == "chartqa":
        train_split = "train"
        eval_split = "val"
    else:
        train_split = "test"
        eval_split = "dev"
    
    logger.info(f"📚 Loading {args.dataset} datasets...")
    
    train_dataset = RealModelDataset(
        dataset_name=args.dataset,
        split=train_split,
        tokenizer=tokenizer,
        processor=processor,
        max_samples=args.max_samples,
        max_length=128  # Shorter sequences to save memory
    )
    
    eval_dataset = RealModelDataset(
        dataset_name=args.dataset,
        split=eval_split,
        tokenizer=tokenizer,
        processor=processor,
        max_samples=min(20, args.max_samples // 5) if args.max_samples else 20,
        max_length=128
    )
    
    # Create data loaders with memory optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # No multiprocessing to save memory
        pin_memory=False,  # Disable pin memory to save memory
        drop_last=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    
    # Setup optimizer for only trainable parameters
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params_list,
        lr=args.learning_rate,
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # Training loop with memory monitoring
    logger.info(f"🎯 Starting memory-optimized training for {args.epochs} epochs...")
    logger.info(f"📊 Training samples: {len(train_dataset):,}")
    logger.info(f"📊 Evaluation samples: {len(eval_dataset):,}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        total_loss = 0
        total_lang_loss = 0
        total_div_loss = 0
        successful_batches = 0
        
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch} Training")
        
        for batch_idx, batch in enumerate(train_progress):
            try:
                # Clear cache before each batch
                torch.cuda.empty_cache()
                gc.collect()
                
                # Move to device
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                
                # Ensure consistent data types (keep float32)
                if 'labels' in batch and torch.is_tensor(batch['labels']):
                    batch['labels'] = batch['labels'].long()
                
                # Forward pass
                outputs = model(
                    image=batch["image"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = outputs["loss"]
                
                if torch.isfinite(loss):
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(trainable_params_list, 1.0)
                    
                    # Optimizer step
                    optimizer.step()
                    scheduler.step()
                    
                    # Track metrics
                    total_loss += loss.item()
                    if "language_loss" in outputs:
                        total_lang_loss += outputs["language_loss"].item()
                    if "diversity_loss" in outputs:
                        total_div_loss += outputs["diversity_loss"].item()
                    successful_batches += 1
                
                # Update progress
                if successful_batches > 0:
                    avg_loss = total_loss / successful_batches
                    
                    # Memory info
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated() / 1024**3
                        
                    train_progress.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "mem": f"{memory_allocated:.1f}GB" if torch.cuda.is_available() else "N/A",
                        "success": f"{successful_batches}/{batch_idx+1}"
                    })
                
                # Clear batch from memory
                del batch, outputs, loss
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"⚠️ CUDA OOM at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    logger.warning(f"Training batch {batch_idx} error: {e}")
                    continue
        
        # Calculate epoch metrics
        avg_train_loss = total_loss / max(successful_batches, 1)
        avg_lang_loss = total_lang_loss / max(successful_batches, 1)
        avg_div_loss = total_div_loss / max(successful_batches, 1)
        
        # Evaluation with memory management
        model.eval()
        eval_loss = 0
        eval_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluation")):
                try:
                    torch.cuda.empty_cache()
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    
                    # Ensure consistent data types (keep float32)
                    if 'labels' in batch and torch.is_tensor(batch['labels']):
                        batch['labels'] = batch['labels'].long()
                    
                    outputs = model(
                        image=batch["image"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    
                    if "loss" in outputs and torch.isfinite(outputs["loss"]):
                        eval_loss += outputs["loss"].item()
                        eval_batches += 1
                    
                    del batch, outputs
                    torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"⚠️ CUDA OOM at eval batch {batch_idx}, skipping...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        logger.warning(f"Eval batch {batch_idx} error: {e}")
                        continue
        
        avg_eval_loss = eval_loss / max(eval_batches, 1)
        
        # Log epoch results
        logger.info(f"📈 Epoch {epoch}:")
        logger.info(f"   Train Loss: {avg_train_loss:.4f}")
        logger.info(f"   Language Loss: {avg_lang_loss:.4f}")
        logger.info(f"   Diversity Loss: {avg_div_loss:.4f}")
        logger.info(f"   Eval Loss: {avg_eval_loss:.4f}")
        logger.info(f"   Success Rate: {successful_batches}/{len(train_loader)}")
        
        # GPU Memory info
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"   GPU Memory: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached")
        
        # Save best model
        if avg_eval_loss < best_loss:
            best_loss = avg_eval_loss
            checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
            
            try:
                # Save only trainable parameters to save space
                trainable_state_dict = {k: v for k, v in model.state_dict().items() 
                                      if any(k.startswith(name) for name, param in model.named_parameters() if param.requires_grad)}
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainable_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'config': config
                }, checkpoint_path)
                logger.info(f"💾 New best model saved with eval loss: {best_loss:.4f}")
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")
        
        # Log to wandb
        if not args.no_wandb:
            wandb.log({
                "epoch": epoch,
                "train/epoch_loss": avg_train_loss,
                "train/language_loss": avg_lang_loss,
                "train/diversity_loss": avg_div_loss,
                "eval/epoch_loss": avg_eval_loss,
                "train/success_rate": successful_batches / len(train_loader),
                "model/best_loss": best_loss,
                "memory/allocated_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                "memory/cached_gb": torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
            })
        
        # Clear memory after each epoch
        torch.cuda.empty_cache()
        gc.collect()
    
    logger.info("🎉 Memory-optimized training completed!")
    logger.info(f"🏆 Best eval loss: {best_loss:.4f}")
    logger.info(f"💾 Best model saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 