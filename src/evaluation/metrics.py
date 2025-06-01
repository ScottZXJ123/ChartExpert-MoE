"""
Evaluation metrics for ChartExpert-MoE

Implements metrics specific to chart reasoning tasks and ChartMuseum benchmark.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import re


class ChartMuseumMetrics:
    """
    Metrics specifically designed for ChartMuseum benchmark evaluation
    
    Handles different reasoning types:
    - Text-dominant reasoning
    - Visual-dominant reasoning  
    - Text/Visual combined reasoning
    - Comprehensive reasoning
    """
    
    def __init__(self):
        self.reasoning_types = [
            'text_dominant',
            'visual_dominant', 
            'text_visual_combined',
            'comprehensive'
        ]
        
        # Error categories from ChartMuseum
        self.error_categories = [
            'symbol_selection',
            'visual_comparison',
            'trajectory_tracking',
            'xy_value_identification',
            'ocr_error',
            'logical_error',
            'hallucination'
        ]
    
    def compute_metrics(
        self,
        predictions: List[str],
        ground_truths: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive metrics for ChartMuseum
        
        Args:
            predictions: Model predictions
            ground_truths: Ground truth answers
            metadata: Optional metadata including reasoning types, chart types, etc.
            
        Returns:
            Dictionary of metrics
        """
        metrics = defaultdict(list)
        
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            # Overall accuracy
            is_correct = self._check_answer(pred, gt)
            metrics['overall_accuracy'].append(is_correct)
            
            # Reasoning type specific accuracy
            if metadata and i < len(metadata):
                reasoning_type = metadata[i].get('reasoning_type', 'unknown')
                chart_type = metadata[i].get('chart_type', 'unknown')
                visual_complexity = metadata[i].get('visual_complexity', 'medium')
                
                metrics[f'accuracy_{reasoning_type}'].append(is_correct)
                metrics[f'accuracy_chart_{chart_type}'].append(is_correct)
                metrics[f'accuracy_complexity_{visual_complexity}'].append(is_correct)
                
                # Error analysis if incorrect
                if not is_correct:
                    error_type = self._classify_error(pred, gt, metadata[i])
                    metrics[f'error_{error_type}'].append(1)
        
        # Aggregate metrics
        aggregated = {}
        for key, values in metrics.items():
            if values:
                aggregated[key] = np.mean(values)
        
        # Add visual reasoning gap metric
        if 'accuracy_visual_dominant' in aggregated and 'accuracy_text_dominant' in aggregated:
            aggregated['visual_reasoning_gap'] = (
                aggregated['accuracy_text_dominant'] - 
                aggregated['accuracy_visual_dominant']
            )
        
        return aggregated
    
    def _check_answer(self, prediction: str, ground_truth: str) -> bool:
        """Check if prediction matches ground truth"""
        # Normalize answers
        pred_norm = self._normalize_answer(prediction)
        gt_norm = self._normalize_answer(ground_truth)
        
        # Exact match
        if pred_norm == gt_norm:
            return True
        
        # Numeric match with tolerance
        if self._is_numeric_match(pred_norm, gt_norm):
            return True
        
        # Semantic similarity (simplified)
        if self._is_semantic_match(pred_norm, gt_norm):
            return True
        
        return False
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        # Convert to lowercase
        answer = answer.lower().strip()
        
        # Remove articles and punctuation
        answer = re.sub(r'\b(a|an|the)\b', '', answer)
        answer = re.sub(r'[^\w\s\.\-]', '', answer)
        
        # Normalize whitespace
        answer = ' '.join(answer.split())
        
        return answer
    
    def _is_numeric_match(self, pred: str, gt: str, tolerance: float = 0.05) -> bool:
        """Check if two numeric answers match within tolerance"""
        pred_nums = re.findall(r'-?\d+\.?\d*', pred)
        gt_nums = re.findall(r'-?\d+\.?\d*', gt)
        
        if pred_nums and gt_nums:
            try:
                pred_val = float(pred_nums[0])
                gt_val = float(gt_nums[0])
                
                # Relative tolerance
                if gt_val != 0:
                    return abs(pred_val - gt_val) / abs(gt_val) < tolerance
                else:
                    return abs(pred_val - gt_val) < tolerance
            except ValueError:
                pass
        
        return False
    
    def _is_semantic_match(self, pred: str, gt: str) -> bool:
        """Simple semantic matching (could be enhanced with embeddings)"""
        # Check for common synonyms
        synonyms = {
            'increase': ['rise', 'grow', 'climb', 'ascend'],
            'decrease': ['fall', 'drop', 'decline', 'descend'],
            'highest': ['maximum', 'peak', 'top', 'greatest'],
            'lowest': ['minimum', 'bottom', 'least', 'smallest']
        }
        
        for base_word, syn_list in synonyms.items():
            if base_word in pred or base_word in gt:
                for syn in syn_list:
                    if (base_word in pred and syn in gt) or (base_word in gt and syn in pred):
                        return True
        
        return False
    
    def _classify_error(self, pred: str, gt: str, metadata: Dict[str, Any]) -> str:
        """Classify the type of error made"""
        # Simple heuristic-based error classification
        
        # Check for hallucination (completely unrelated answer)
        if len(set(pred.split()) & set(gt.split())) == 0:
            return 'hallucination'
        
        # Check for OCR error (if numbers are close but not exact)
        if self._is_numeric_match(pred, gt, tolerance=0.2) and not self._is_numeric_match(pred, gt, tolerance=0.05):
            return 'ocr_error'
        
        # Visual-specific errors based on metadata
        if metadata.get('requires_visual_comparison', False):
            return 'visual_comparison'
        
        if metadata.get('requires_trajectory_tracking', False):
            return 'trajectory_tracking'
        
        # Default to logical error
        return 'logical_error'


class ChartMetrics:
    """General metrics for chart reasoning tasks"""
    
    def __init__(self):
        self.metrics_computer = ChartMuseumMetrics()
    
    def compute_accuracy(self, predictions: List[str], ground_truths: List[str]) -> float:
        """Compute simple accuracy"""
        correct = sum(
            self.metrics_computer._check_answer(pred, gt) 
            for pred, gt in zip(predictions, ground_truths)
        )
        return correct / len(predictions) if predictions else 0.0
    
    def compute_f1_score(self, predictions: List[str], ground_truths: List[str]) -> float:
        """Compute F1 score for extraction tasks"""
        # Simplified F1 calculation
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for pred, gt in zip(predictions, ground_truths):
            pred_tokens = set(pred.lower().split())
            gt_tokens = set(gt.lower().split())
            
            true_positives += len(pred_tokens & gt_tokens)
            false_positives += len(pred_tokens - gt_tokens)
            false_negatives += len(gt_tokens - pred_tokens)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1
    
    def compute_expert_activation_metrics(
        self,
        expert_activations: List[Dict[str, float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """Analyze expert activation patterns"""
        metrics = defaultdict(list)
        
        for i, activation in enumerate(expert_activations):
            # Average activation per expert
            for expert_name, activation_value in activation.items():
                metrics[f'avg_activation_{expert_name}'].append(activation_value)
            
            # Activation patterns by reasoning type
            if metadata and i < len(metadata):
                reasoning_type = metadata[i].get('reasoning_type', 'unknown')
                for expert_name, activation_value in activation.items():
                    metrics[f'activation_{expert_name}_{reasoning_type}'].append(activation_value)
        
        # Aggregate
        aggregated = {}
        for key, values in metrics.items():
            if values:
                aggregated[key] = np.mean(values)
        
        return aggregated 