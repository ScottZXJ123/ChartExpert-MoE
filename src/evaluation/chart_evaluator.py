"""
Chart-specific evaluation for ChartExpert-MoE

Implements evaluation logic for chart reasoning tasks including ChartMuseum benchmark.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from collections import defaultdict
import re
from tqdm import tqdm


class ChartEvaluator:
    """Evaluator for chart reasoning tasks"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.metrics = ChartMetrics()
        
    def evaluate_chartmuseum(
        self,
        dataset,
        batch_size: int = 8,
        reasoning_types: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate on ChartMuseum benchmark
        
        Args:
            dataset: ChartMuseum dataset
            batch_size: Batch size for evaluation
            reasoning_types: Optional list of reasoning types to evaluate
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        results = defaultdict(list)
        
        # Group by reasoning type if specified
        if reasoning_types:
            dataset = dataset.filter(lambda x: x['reasoning_type'] in reasoning_types)
        
        with torch.no_grad():
            for batch in self._create_batches(dataset, batch_size):
                predictions = self._get_predictions(batch)
                
                for i, pred in enumerate(predictions):
                    ground_truth = batch['answers'][i]
                    reasoning_type = batch.get('reasoning_type', [None])[i]
                    
                    # Evaluate different aspects
                    results['overall'].append(
                        self.metrics.compute_accuracy(pred, ground_truth)
                    )
                    
                    if reasoning_type:
                        results[f'reasoning_{reasoning_type}'].append(
                            self.metrics.compute_accuracy(pred, ground_truth)
                        )
        
        # Aggregate results
        aggregated = {}
        for key, values in results.items():
            aggregated[key] = np.mean(values)
            
        return aggregated
    
    def evaluate_chartqa(self, dataset, batch_size: int = 8) -> Dict[str, float]:
        """Evaluate on ChartQA dataset"""
        return self._evaluate_vqa_dataset(dataset, batch_size, "chartqa")
    
    def evaluate_plotqa(self, dataset, batch_size: int = 8) -> Dict[str, float]:
        """Evaluate on PlotQA dataset"""
        return self._evaluate_vqa_dataset(dataset, batch_size, "plotqa")
    
    def _evaluate_vqa_dataset(
        self,
        dataset,
        batch_size: int,
        dataset_name: str
    ) -> Dict[str, float]:
        """Generic VQA evaluation"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self._create_batches(dataset, batch_size):
                predictions = self._get_predictions(batch)
                
                for pred, gt in zip(predictions, batch['answers']):
                    if self.metrics.compute_accuracy(pred, gt):
                        correct += 1
                    total += 1
        
        return {
            f"{dataset_name}_accuracy": correct / total if total > 0 else 0
        }
    
    def _get_predictions(self, batch: Dict[str, Any]) -> List[str]:
        """Get model predictions for a batch"""
        # This is a simplified version - in practice would use actual model inference
        outputs = self.model(
            image=batch['image'].to(self.device),
            input_ids=batch['input_ids'].to(self.device),
            attention_mask=batch['attention_mask'].to(self.device)
        )
        
        # Decode predictions
        predictions = []
        logits = outputs['logits']
        
        # Simple greedy decoding (in practice, use proper generation)
        pred_ids = torch.argmax(logits, dim=-1)
        
        # Mock decoding for now
        for i in range(len(batch['image'])):
            predictions.append("mock_prediction")  # Would use tokenizer.decode
            
        return predictions
    
    def _create_batches(self, dataset, batch_size: int):
        """Create batches from dataset"""
        # Simplified batching - in practice would use DataLoader
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            yield self._prepare_batch(batch)
    
    def _prepare_batch(self, samples: List[Dict]) -> Dict[str, Any]:
        """Prepare a batch for evaluation"""
        # Mock batch preparation
        batch_size = len(samples)
        return {
            'image': torch.randn(batch_size, 3, 224, 224),  # Mock images
            'input_ids': torch.randint(0, 1000, (batch_size, 100)),  # Mock tokens
            'attention_mask': torch.ones(batch_size, 100),
            'answers': [s.get('answer', '') for s in samples],
            'reasoning_type': [s.get('reasoning_type', 'unknown') for s in samples]
        }


class ChartMetrics:
    """Metrics for chart reasoning evaluation"""
    
    def compute_accuracy(self, prediction: str, ground_truth: str) -> bool:
        """
        Compute accuracy for chart QA
        
        Handles different answer formats (numeric, text, etc.)
        """
        # Normalize both strings
        pred_norm = self._normalize_answer(prediction)
        gt_norm = self._normalize_answer(ground_truth)
        
        # Exact match
        if pred_norm == gt_norm:
            return True
            
        # Numeric comparison with tolerance
        pred_num = self._extract_number(pred_norm)
        gt_num = self._extract_number(gt_norm)
        
        if pred_num is not None and gt_num is not None:
            return abs(pred_num - gt_num) / (abs(gt_num) + 1e-5) < 0.05  # 5% tolerance
            
        return False
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer string"""
        # Convert to lowercase
        answer = answer.lower().strip()
        
        # Remove articles
        answer = re.sub(r'\b(a|an|the)\b', '', answer)
        
        # Remove punctuation
        answer = re.sub(r'[^\w\s\.]', '', answer)
        
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        
        return answer
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numeric value from text"""
        # Find all numbers in the text
        numbers = re.findall(r'-?\d+\.?\d*', text)
        
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                return None
        return None 