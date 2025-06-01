#!/usr/bin/env python3
"""
Training script for ChartExpert-MoE

This script handles the multi-stage training process for ChartExpert-MoE,
including foundation pre-training, joint pre-training, chart-specific tuning,
expert specialization, and ChartMuseum fine-tuning.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import wandb
from tqdm import tqdm
import logging
from typing import Dict, Any, Optional

# Import from the installed package
from models import ChartExpertMoE
from data import ChartMuseumDataset, ChartQADataset, MultiDatasetLoader
from training import MultiStageTrainer
from evaluation import ChartEvaluator
from utils import setup_logging, save_config, load_config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train ChartExpert-MoE")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["foundation", "joint_pretrain", "chart_tuning", "expert_specialization", "chartmuseum_finetune", "all"],
        help="Training stage to run"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints"
    )
    
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with reduced data"
    )
    
    return parser.parse_args()


def setup_distributed():
    """Setup distributed training if available"""
    if torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return local_rank, True
    return 0, False


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup distributed training
    local_rank, distributed = setup_distributed()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    setup_logging(
        log_level=config.get("logging", {}).get("log_level", "INFO"),
        log_dir=config.get("logging", {}).get("log_dir", "./logs")
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting ChartExpert-MoE training - Stage: {args.stage}")
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        logger.info(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Initialize wandb if enabled
    if config.get("logging", {}).get("wandb", {}).get("enabled", False) and local_rank == 0:
        wandb.init(
            project=config["logging"]["wandb"]["project"],
            entity=config["logging"]["wandb"].get("entity"),
            config=config,
            name=f"chartexpert_moe_{args.stage}"
        )
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["llm_backbone"]["model_name"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    logger.info("Initializing ChartExpert-MoE model...")
    model = ChartExpertMoE(config)
    
    # Load from checkpoint if resuming
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # Move model to device
    model = model.to(device)
    
    # Setup distributed training
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            output_device=local_rank
        )
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Setup data
    logger.info("Setting up datasets...")
    data_loader = MultiDatasetLoader(tokenizer, config["data"])
    
    # Initialize trainer
    trainer = MultiStageTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
        local_rank=local_rank,
        output_dir=args.output_dir
    )
    
    # Run training stages
    if args.stage == "all":
        stages = ["foundation", "joint_pretrain", "chart_tuning", "expert_specialization", "chartmuseum_finetune"]
    else:
        stages = [args.stage]
    
    for stage in stages:
        logger.info(f"Starting stage: {stage}")
        
        # Get stage-specific configuration
        stage_config = config["training"]["stages"][stage]
        
        # Setup datasets for this stage
        if stage == "foundation":
            # Use general vision-language data (placeholder)
            train_dataset = None  # Would load general VL data
            val_dataset = None
        elif stage == "joint_pretrain":
            # Use mixed chart and general data
            train_dataset = data_loader.get_combined_dataset(["chartqa"], "train")
            val_dataset = data_loader.get_combined_dataset(["chartqa"], "validation")
        elif stage == "chart_tuning":
            # Use chart-specific data
            train_dataset = data_loader.get_combined_dataset(["chartqa", "plotqa"], "train")
            val_dataset = data_loader.get_combined_dataset(["chartqa"], "validation")
        elif stage == "expert_specialization":
            # Use diverse chart data for expert specialization
            train_dataset = data_loader.get_combined_dataset(["chartqa"], "train")
            val_dataset = data_loader.get_combined_dataset(["chartqa"], "validation")
        elif stage == "chartmuseum_finetune":
            # Use ChartMuseum for final fine-tuning
            train_dataset = data_loader.load_chartmuseum("train") if hasattr(data_loader.load_chartmuseum("test").dataset, "train") else None
            val_dataset = data_loader.load_chartmuseum("test")
        
        # Skip if no training data for this stage
        if train_dataset is None:
            logger.warning(f"No training data for stage {stage}, skipping...")
            continue
        
        # Create data loaders
        if args.debug:
            # Use smaller datasets for debugging
            train_dataset = torch.utils.data.Subset(train_dataset, range(min(100, len(train_dataset))))
            if val_dataset:
                val_dataset = torch.utils.data.Subset(val_dataset, range(min(50, len(val_dataset))))
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=stage_config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=stage_config["batch_size"],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Train this stage
        trainer.train_stage(
            stage=stage,
            train_loader=train_loader,
            val_loader=val_loader,
            stage_config=stage_config
        )
        
        logger.info(f"Completed stage: {stage}")
    
    # Final evaluation on ChartMuseum
    if local_rank == 0:
        logger.info("Running final evaluation on ChartMuseum...")
        
        evaluator = ChartEvaluator(config["evaluation"])
        chartmuseum_dataset = data_loader.load_chartmuseum("test")
        
        eval_loader = DataLoader(
            chartmuseum_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2
        )
        
        results = evaluator.evaluate(
            model=model.module if distributed else model,
            tokenizer=tokenizer,
            data_loader=eval_loader,
            device=device
        )
        
        logger.info("Final ChartMuseum Results:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Log to wandb
        if config.get("logging", {}).get("wandb", {}).get("enabled", False):
            wandb.log({"final_evaluation": results})
    
    # Save final model
    if local_rank == 0:
        final_save_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_save_path, exist_ok=True)
        
        model_to_save = model.module if distributed else model
        model_to_save.save_pretrained(final_save_path)
        
        # Save configuration
        save_config(config, os.path.join(final_save_path, "training_config.yaml"))
        
        logger.info(f"Final model saved to {final_save_path}")
    
    # Cleanup
    if config.get("logging", {}).get("wandb", {}).get("enabled", False) and local_rank == 0:
        wandb.finish()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main() 