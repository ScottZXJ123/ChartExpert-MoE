"""
Minimal ChartEvaluator implementation for training
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any
import logging
from tqdm import tqdm


class ChartEvaluator:
    """Basic evaluator for ChartExpert-MoE during training"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def evaluate(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        data_loader: DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """Basic evaluation that computes loss and simple metrics"""
        model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                
                if "loss" in outputs:
                    total_loss += outputs["loss"].item()
                    total_samples += 1
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        # Basic metrics
        results = {
            "average_loss": avg_loss,
            "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
            "num_samples": total_samples
        }
        
        return results 