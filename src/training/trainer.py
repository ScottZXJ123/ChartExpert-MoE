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
import torch.optim as optim
from transformers import get_scheduler

from models import ChartExpertMoE
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
        optimizer = optim.AdamW(self.model.parameters(), lr=stage_config["learning_rate"], weight_decay=stage_config.get("weight_decay", 0.01))
        
        total_steps = len(train_loader) * stage_config["epochs"]
        scheduler = get_scheduler(
            name=stage_config.get("scheduler", "cosine_with_restarts"),
            optimizer=optimizer,
            num_warmup_steps=stage_config.get("warmup_steps", int(0.1 * total_steps)),
            num_training_steps=total_steps
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

    def train_stage(self, stage_name: str, train_loader: DataLoader, val_loader: Optional[DataLoader], stage_config: Dict[str, Any]):
        """
        Trains the model for a specific stage.
        """
        self.logger.info(f"Initializing training for stage: {stage_name}")
        
        # Optimizer
        optimizer_config = stage_config.get("optimizer", {})
        learning_rate = optimizer_config.get("lr", 1e-4)
        weight_decay = optimizer_config.get("weight_decay", 0.01)
        
        # Filter out parameters that don't require gradients
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            self.logger.warning(f"No trainable parameters found for stage {stage_name}. Skipping optimizer setup.")
            # Potentially skip the stage or just run evaluation if val_loader is present
            if val_loader and self.local_rank in [-1, 0]:
                self._evaluate_stage(stage_name, val_loader, stage_config)
            return

        optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

        # Scheduler
        scheduler_config = stage_config.get("scheduler", {})
        num_epochs = stage_config.get("num_epochs", 3)
        
        # Calculate total training steps based on actual loader size
        # Handle cases where train_loader might be a DistributedSampler wrapped loader
        try:
            num_training_steps_per_epoch = len(train_loader)
        except TypeError: # Happens if train_loader doesn't have a __len__ (e.g. IterableDataset)
            # Estimate steps if loader length is not available, or require it in config
            num_training_steps_per_epoch = stage_config.get("steps_per_epoch_estimate", 1000) 
            self.logger.warning(f"Length of train_loader for stage {stage_name} not available. Using estimate: {num_training_steps_per_epoch} steps/epoch.")

        num_training_steps = num_epochs * num_training_steps_per_epoch
        
        # Default warmup to 10% of total steps if not specified or if set to a fraction
        warmup_setting = scheduler_config.get("warmup", 0.1)
        if isinstance(warmup_setting, float) and 0 <= warmup_setting <= 1:
            num_warmup_steps = int(warmup_setting * num_training_steps)
        elif isinstance(warmup_setting, int):
            num_warmup_steps = warmup_setting
        else: # Default to 0 if invalid
            num_warmup_steps = 0
            
        scheduler_type = scheduler_config.get("type", "cosine_with_restarts")
        
        # Handle case where num_training_steps might be 0 (e.g. empty dataloader)
        if num_training_steps == 0 :
            self.logger.warning(f"Number of training steps is 0 for stage {stage_name}. Skipping scheduler setup and training loop.")
            # Potentially run evaluation
            if val_loader and self.local_rank in [-1, 0]:
                self._evaluate_stage(stage_name, val_loader, stage_config)
            return

        scheduler = get_scheduler(
            name=scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        start_epoch = 0
        # TODO: Add logic for resuming from a stage-specific checkpoint (load optimizer, scheduler states)

        self.logger.info(f"  Optimizer: AdamW, LR: {learning_rate}, Weight Decay: {weight_decay}")
        self.logger.info(f"  Scheduler: {scheduler_type}, Warmup Steps: {num_warmup_steps}, Total Training Steps: {num_training_steps}")
        self.logger.info(f"  Num Epochs: {num_epochs}, Steps per Epoch: {num_training_steps_per_epoch}")

        best_val_metric = float('inf') # Lower is better for loss

        for epoch in range(start_epoch, num_epochs):
            self.logger.info(f"--- Stage {stage_name}, Epoch {epoch+1}/{num_epochs} ---")
            self.model.train()
            total_train_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", disable=(self.local_rank not in [-1, 0]))
            
            for batch_idx, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                outputs = self.model(**batch)
                loss = outputs.get("loss")
                
                if loss is None:
                    self.logger.error(f"Loss not found in model outputs for stage {stage_name}. Skipping batch.")
                    continue

                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.warning(f"NaN or Inf loss detected at batch {batch_idx} in stage {stage_name}. Skipping batch.")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, stage_config.get("max_grad_norm", 1.0))
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
                
                if self.local_rank in [-1, 0] and batch_idx % stage_config.get("log_interval", 10) == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    log_data = {
                        f"train/{stage_name}/loss_step": loss.item(),
                        f"train/{stage_name}/lr": current_lr,
                        f"train/{stage_name}/epoch_progress": epoch + (batch_idx / num_training_steps_per_epoch)
                    }
                    if "aux_loss" in outputs and outputs["aux_loss"] is not None:
                         log_data[f"train/{stage_name}/aux_loss_step"] = outputs["aux_loss"].item()
                    if wandb.run:
                        wandb.log(log_data)
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

            avg_train_loss = total_train_loss / num_training_steps_per_epoch if num_training_steps_per_epoch > 0 else 0.0
            self.logger.info(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")
            if self.local_rank in [-1, 0] and wandb.run:
                 wandb.log({f"train/{stage_name}/loss_epoch": avg_train_loss, "epoch": epoch + 1, f"epoch_{stage_name}": epoch + 1})

            if val_loader and self.local_rank in [-1, 0]:
                val_metrics = self._evaluate_stage(stage_name, val_loader, stage_config)
                # Default to monitor validation loss, assuming lower is better.
                metric_to_monitor = stage_config.get("eval_metric_to_monitor", f"val/{stage_name}/loss")
                current_val_metric = val_metrics.get(metric_to_monitor, float('inf'))

                if wandb.run:
                    wandb.log({**val_metrics, "epoch": epoch + 1, f"epoch_{stage_name}": epoch + 1})
                
                save_checkpoint_flag = False
                is_current_best = False
                if stage_config.get("save_best_checkpoint", True):
                    # Assuming lower is better for the monitored metric (e.g., loss)
                    if current_val_metric < best_val_metric:
                        best_val_metric = current_val_metric
                        save_checkpoint_flag = True
                        is_current_best = True
                        self.logger.info(f"New best validation metric ({metric_to_monitor}): {best_val_metric:.4f}. Saving model...")
                elif stage_config.get("save_every_epoch", False): # Option to save every epoch if not just best
                    save_checkpoint_flag = True
                
                if save_checkpoint_flag:
                    self._save_checkpoint(stage_name, epoch, optimizer, scheduler, is_best=is_current_best)
            elif self.local_rank in [-1, 0] and stage_config.get("save_every_epoch", True): # No val_loader, but save_every_epoch is true
                 self._save_checkpoint(stage_name, epoch, optimizer, scheduler, is_best=False)

        self.logger.info(f"Finished training stage: {stage_name}")

    def _evaluate_stage(self, stage_name: str, val_loader: DataLoader, stage_config: Dict[str, Any]) -> Dict[str, float]:
        self.model.eval()
        total_val_loss = 0.0
        # TODO: Initialize accumulators for other metrics
        
        self.logger.info(f"Starting validation for stage: {stage_name}...")
        progress_bar = tqdm(val_loader, desc=f"Stage {stage_name} Validation", disable=(self.local_rank not in [-1, 0]))

        with torch.no_grad():
            for batch in progress_bar:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.get("loss")
                if loss is not None:
                    total_val_loss += loss.item()
                # TODO: Calculate and accumulate other eval metrics based on model output and batch
                # e.g., accuracy, F1, perplexity. This might require an Evaluator class.

        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        self.logger.info(f"Stage {stage_name} Validation Loss: {avg_val_loss:.4f}")
        
        metrics = {
            f"val/{stage_name}/loss": avg_val_loss,
            # TODO: Add other computed metrics here
            # metrics[f"val/{stage_name}/accuracy"] = computed_accuracy
        }
        return metrics

    def _save_checkpoint(self, stage_name: str, epoch: int, optimizer, scheduler, is_best: bool = False):
        if self.local_rank not in [-1, 0]:
            return

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': self.config,
            'stage_name': stage_name,
            'tokenizer_name_or_path': self.tokenizer.name_or_path if hasattr(self.tokenizer, 'name_or_path') else None
        }
        
        # Stage-specific save directory
        stage_save_dir = os.path.join(self.output_dir, stage_name)
        os.makedirs(stage_save_dir, exist_ok=True)
        
        filename_prefix = f"checkpoint_stage_{stage_name}"
        epoch_filename = f"{filename_prefix}_epoch_{epoch+1}.pt"
        full_epoch_save_path = os.path.join(stage_save_dir, epoch_filename)

        # Save epoch checkpoint
        torch.save(checkpoint_data, full_epoch_save_path)
        self.logger.info(f"Saved epoch checkpoint to {full_epoch_save_path}")

        if is_best:
            best_filename = f"{filename_prefix}_best.pt"
            full_best_save_path = os.path.join(stage_save_dir, best_filename)
            torch.save(checkpoint_data, full_best_save_path) # Could also symlink
            self.logger.info(f"Saved best checkpoint to {full_best_save_path}")
        
        # Save tokenizer (can be large if it's a custom/modified one, often not needed if using from_pretrained name)
        # self.tokenizer.save_pretrained(os.path.join(stage_save_dir, f"tokenizer_epoch_{epoch+1}"))
        # if is_best:
        #     self.tokenizer.save_pretrained(os.path.join(stage_save_dir, "tokenizer_best"))

# It's good practice to have an __all__ in __init__.py if you have one.
# If src/training/__init__.py doesn't exist, it should be created.
# Content for src/training/__init__.py:
# from .trainer import MultiStageTrainer
# __all__ = ["MultiStageTrainer"] 