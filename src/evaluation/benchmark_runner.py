"""
Benchmark runner for ChartExpert-MoE

Coordinates evaluation across different benchmarks and generates comprehensive reports.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import json
import os
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import numpy as np

from .chart_evaluator import ChartEvaluator
from .metrics import ChartMuseumMetrics, ChartMetrics


class BenchmarkRunner:
    """Run comprehensive benchmark evaluations"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: str = "./benchmark_results"
    ):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.evaluator = ChartEvaluator(model, device)
        
        os.makedirs(output_dir, exist_ok=True)
    
    def run_chartmuseum_benchmark(
        self,
        dataset,
        batch_size: int = 8,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive ChartMuseum benchmark
        
        Evaluates:
        - Overall accuracy
        - Accuracy by reasoning type
        - Accuracy by visual complexity
        - Error analysis
        - Expert activation patterns
        """
        print("Running ChartMuseum Benchmark...")
        
        results = {
            'benchmark': 'ChartMuseum',
            'timestamp': datetime.now().isoformat(),
            'model_config': self._get_model_config(),
            'metrics': {}
        }
        
        # Evaluate by reasoning type
        reasoning_types = [
            'text_dominant',
            'visual_dominant',
            'text_visual_combined', 
            'comprehensive'
        ]
        
        for reasoning_type in reasoning_types:
            print(f"Evaluating {reasoning_type} reasoning...")
            type_metrics = self.evaluator.evaluate_chartmuseum(
                dataset,
                batch_size=batch_size,
                reasoning_types=[reasoning_type]
            )
            results['metrics'].update(type_metrics)
        
        # Overall evaluation
        print("Evaluating overall performance...")
        overall_metrics = self.evaluator.evaluate_chartmuseum(
            dataset,
            batch_size=batch_size
        )
        results['metrics'].update(overall_metrics)
        
        # Calculate visual reasoning gap
        if 'reasoning_visual_dominant' in results['metrics'] and 'reasoning_text_dominant' in results['metrics']:
            results['metrics']['visual_reasoning_gap'] = (
                results['metrics']['reasoning_text_dominant'] - 
                results['metrics']['reasoning_visual_dominant']
            )
        
        # Save results
        if save_results:
            self._save_results(results, 'chartmuseum')
        
        return results
    
    def run_all_benchmarks(
        self,
        datasets: Dict[str, Any],
        batch_size: int = 8
    ) -> Dict[str, Dict[str, Any]]:
        """Run evaluation on all available benchmarks"""
        all_results = {}
        
        # ChartMuseum
        if 'chartmuseum' in datasets:
            all_results['chartmuseum'] = self.run_chartmuseum_benchmark(
                datasets['chartmuseum'],
                batch_size=batch_size
            )
        
        # ChartQA
        if 'chartqa' in datasets:
            print("Running ChartQA Benchmark...")
            chartqa_results = self.evaluator.evaluate_chartqa(
                datasets['chartqa'],
                batch_size=batch_size
            )
            all_results['chartqa'] = {
                'benchmark': 'ChartQA',
                'metrics': chartqa_results
            }
        
        # PlotQA
        if 'plotqa' in datasets:
            print("Running PlotQA Benchmark...")
            plotqa_results = self.evaluator.evaluate_plotqa(
                datasets['plotqa'],
                batch_size=batch_size
            )
            all_results['plotqa'] = {
                'benchmark': 'PlotQA',
                'metrics': plotqa_results
            }
        
        # Generate comparative report
        self._generate_comparative_report(all_results)
        
        return all_results
    
    def analyze_expert_activation_patterns(
        self,
        dataset,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """Analyze how different experts are activated for different types of questions"""
        print("Analyzing expert activation patterns...")
        
        activation_data = []
        
        # Sample from dataset
        samples = dataset.select(range(min(num_samples, len(dataset))))
        
        for sample in tqdm(samples, desc="Analyzing activations"):
            # Get model prediction with expert activations
            result = self.model.predict(
                image_path=sample.get('image_path', ''),
                query=sample.get('question', '')
            )
            
            activation_data.append({
                'reasoning_type': sample.get('reasoning_type', 'unknown'),
                'chart_type': sample.get('chart_type', 'unknown'),
                'expert_activations': result.get('expert_activations', {}),
                'is_correct': result.get('response', '') == sample.get('answer', '')
            })
        
        # Analyze patterns
        analysis = self._analyze_activation_data(activation_data)
        
        # Save analysis
        self._save_activation_analysis(analysis)
        
        return analysis
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Extract model configuration"""
        config = {
            'model_type': 'ChartExpert-MoE',
            'num_experts': len(getattr(self.model, 'experts', [])),
            'vision_encoder': getattr(self.model, 'vision_encoder', {}).get('type', 'unknown'),
            'llm_backbone': getattr(self.model, 'llm_backbone', {}).get('type', 'unknown')
        }
        return config
    
    def _save_results(self, results: Dict[str, Any], benchmark_name: str):
        """Save evaluation results"""
        filename = f"{benchmark_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def _generate_comparative_report(self, all_results: Dict[str, Dict[str, Any]]):
        """Generate a comparative report across benchmarks"""
        report_data = []
        
        for benchmark, results in all_results.items():
            metrics = results.get('metrics', {})
            report_data.append({
                'benchmark': benchmark,
                'overall_accuracy': metrics.get('overall', 0),
                'visual_reasoning_gap': metrics.get('visual_reasoning_gap', 0)
            })
        
        # Create DataFrame for easy viewing
        df = pd.DataFrame(report_data)
        
        # Save as CSV
        csv_path = os.path.join(
            self.output_dir,
            f"comparative_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        df.to_csv(csv_path, index=False)
        
        print("\nComparative Report:")
        print(df.to_string())
        print(f"\nReport saved to {csv_path}")
    
    def _analyze_activation_data(self, activation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze expert activation patterns"""
        analysis = {
            'by_reasoning_type': {},
            'by_correctness': {'correct': {}, 'incorrect': {}},
            'overall_patterns': {}
        }
        
        # Group by reasoning type
        from collections import defaultdict
        by_reasoning = defaultdict(list)
        
        for item in activation_data:
            reasoning_type = item['reasoning_type']
            by_reasoning[reasoning_type].append(item['expert_activations'])
            
            # By correctness
            correctness = 'correct' if item['is_correct'] else 'incorrect'
            for expert, activation in item['expert_activations'].items():
                if expert not in analysis['by_correctness'][correctness]:
                    analysis['by_correctness'][correctness][expert] = []
                analysis['by_correctness'][correctness][expert].append(activation)
        
        # Compute averages
        for reasoning_type, activations_list in by_reasoning.items():
            avg_activations = {}
            if activations_list:
                all_experts = set()
                for act in activations_list:
                    all_experts.update(act.keys())
                
                for expert in all_experts:
                    values = [act.get(expert, 0) for act in activations_list]
                    avg_activations[expert] = np.mean(values)
            
            analysis['by_reasoning_type'][reasoning_type] = avg_activations
        
        return analysis
    
    def _save_activation_analysis(self, analysis: Dict[str, Any]):
        """Save expert activation analysis"""
        filename = f"activation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Activation analysis saved to {filepath}") 