"""
Trainer classes for ChartExpert-MoE

Implements the core training logic including multi-stage training support.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Optional, Any, List, Tuple
import wandb
from tqdm import tqdm
import logging
import os
from collections import defaultdict

from ..models import ChartExpertMoE
from .loss_functions import ChartMoELoss


class Trainer:
    """Base trainer class for ChartExpert-MoE"""
    
    def __init__(
        self,
        model: ChartExpertMoE,
        config: Dict[str, Any],
        device: torch.device,
        output_dir: str = "./checkpoints"
    ):
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('-inf')
        
        # Loss function
        self.criterion = ChartMoELoss(config)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        accumulation_steps: int = 1
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                image=batch["image"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch.get("labels")
            )
            
            # Calculate loss
            loss = outputs["loss"] / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if self.config.get("gradient_clip_norm", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["gradient_clip_norm"]
                    )
                
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                self.global_step += 1
            
            # Track losses
            epoch_losses["total_loss"] += loss.item() * accumulation_steps
            epoch_losses["lm_loss"] += outputs.get("lm_loss", 0).item()
            epoch_losses["aux_loss"] += outputs.get("aux_loss", 0).item()
            num_batches += 1
            
            # Update progress bar
            avg_loss = epoch_losses["total_loss"] / num_batches
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
            # Log to wandb
            if self.global_step % self.config.get("log_interval", 100) == 0:
                if wandb.run is not None:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lm_loss": epoch_losses["lm_loss"] / num_batches,
                        "train/aux_loss": epoch_losses["aux_loss"] / num_batches,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/global_step": self.global_step
                    })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return dict(epoch_losses)
    
    def evaluate(
        self,
        eval_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        eval_losses = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    image=batch["image"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch.get("labels")
                )
                
                # Track losses
                eval_losses["total_loss"] += outputs["loss"].item()
                eval_losses["lm_loss"] += outputs.get("lm_loss", 0).item()
                eval_losses["aux_loss"] += outputs.get("aux_loss", 0).item()
                num_batches += 1
        
        # Average losses
        for key in eval_losses:
            eval_losses[key] /= num_batches
        
        return dict(eval_losses)
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "metrics": metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.output_dir, 
            f"checkpoint_epoch_{self.epoch}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")


class MultiStageTrainer(Trainer):
    """Multi-stage trainer for ChartExpert-MoE"""
    
    def __init__(
        self,
        model: ChartExpertMoE,
        tokenizer: Any,
        config: Dict[str, Any],
        device: torch.device,
        local_rank: int = 0,
        output_dir: str = "./checkpoints"
    ):
        super().__init__(model, config, device, output_dir)
        self.tokenizer = tokenizer
        self.local_rank = local_rank
        self.stages = ["foundation", "joint_pretrain", "chart_tuning", 
                      "expert_specialization", "chartmuseum_finetune"]
    
    def train_stage(
        self,
        stage: str,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        stage_config: Dict[str, Any]
    ):
        """Train a specific stage"""
        self.logger.info(f"Starting training stage: {stage}")
        
        # Create optimizer and scheduler for this stage
        from .optimizer_utils import create_optimizer, create_scheduler
        optimizer = create_optimizer(
            self.model, 
            stage_config["learning_rate"],
            self.config.get("optimizer", "adamw"),
            self.config.get("weight_decay", 0.01)
        )
        
        total_steps = len(train_loader) * stage_config["epochs"]
        scheduler = create_scheduler(
            optimizer,
            self.config.get("scheduler", "cosine"),
            total_steps,
            stage_config.get("warmup_steps", 0)
        )
        
        # Stage-specific training
        best_val_loss = float('inf')
        
        for epoch in range(stage_config["epochs"]):
            self.epoch = epoch
            self.logger.info(f"Stage {stage} - Epoch {epoch + 1}/{stage_config['epochs']}")
            
            # Train
            train_metrics = self.train_epoch(
                train_loader, 
                optimizer, 
                scheduler,
                accumulation_steps=self.config.get("gradient_accumulation_steps", 1)
            )
            
            self.logger.info(f"Train metrics: {train_metrics}")
            
            # Evaluate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self.logger.info(f"Val metrics: {val_metrics}")
                
                # Save best model
                if val_metrics["total_loss"] < best_val_loss:
                    best_val_loss = val_metrics["total_loss"]
                    self.save_checkpoint(val_metrics, is_best=True)
                
                # Log to wandb
                if wandb.run is not None and self.local_rank == 0:
                    wandb.log({
                        f"{stage}/val_loss": val_metrics["total_loss"],
                        f"{stage}/epoch": epoch
                    })
            
            # Save checkpoint
            if (epoch + 1) % self.config.get("save_every_n_epochs", 1) == 0:
                self.save_checkpoint(train_metrics)
        
        self.logger.info(f"Completed training stage: {stage}")
    
    def freeze_components(self, stage: str):
        """Freeze model components based on training stage"""
        if stage == "foundation":
            # Freeze nothing, train everything
            pass
        elif stage == "joint_pretrain":
            # Optionally freeze vision encoder
            if self.config.get("freeze_vision_encoder", False):
                for param in self.model.vision_encoder.parameters():
                    param.requires_grad = False
        elif stage == "chart_tuning":
            # Freeze base models, train experts and fusion
            for param in self.model.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.model.llm_backbone.encoder.parameters():
                param.requires_grad = False
        elif stage == "expert_specialization":
            # Only train expert modules
            for name, param in self.model.named_parameters():
                if "expert" not in name:
                    param.requires_grad = False
        elif stage == "chartmuseum_finetune":
            # Fine-tune everything with small learning rate
            pass 