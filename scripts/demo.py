#!/usr/bin/env python
"""
ChartExpert-MoE Demo Script

This script demonstrates the capabilities of ChartExpert-MoE on chart reasoning tasks.
It can work with mock data for testing or real chart images when available.
"""

import argparse
import torch
import yaml
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import ChartExpertMoE
from src.utils import load_config, setup_logging


def create_mock_model(config: Dict[str, Any]) -> ChartExpertMoE:
    """Create a mock model for demonstration"""
    # For demo purposes, we'll create a smaller config
    demo_config = {
        "hidden_size": 768,
        "vocab_size": 32000,
        "vision_encoder": {
            "type": "clip",
            "model_name": "openai/clip-vit-base-patch32",
            "hidden_size": 768
        },
        "llm_backbone": {
            "type": "llama",
            "model_name": "meta-llama/Llama-2-7b-hf",
            "hidden_size": 4096
        },
        "experts": config["experts"],
        "routing": config["routing"],
        "moe": config["moe"],
        "fusion": config["fusion"]
    }
    
    # Create model
    model = ChartExpertMoE(demo_config)
    return model


def create_mock_chart_image() -> Image.Image:
    """Create a mock chart image for demonstration"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create sample data
    categories = ['Q1', 'Q2', 'Q3', 'Q4']
    values_2022 = [45, 52, 48, 58]
    values_2023 = [50, 55, 53, 62]
    
    x = np.arange(len(categories))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, values_2022, width, label='2022', color='skyblue')
    bars2 = ax.bar(x + width/2, values_2023, width, label='2023', color='lightcoral')
    
    # Add labels and title
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Sales (in millions)')
    ax.set_title('Quarterly Sales Comparison 2022 vs 2023')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Convert to PIL Image
    fig.canvas.draw()
    img = Image.frombytes('RGB', fig.canvas.get_width_height(), 
                         fig.canvas.tostring_rgb())
    plt.close()
    
    return img


def analyze_expert_activations(routing_weights: torch.Tensor) -> Dict[str, float]:
    """Analyze expert activation patterns"""
    expert_names = [
        "layout", "ocr", "scale", "geometric", "trend",
        "query", "numerical", "integration", "alignment", "orchestrator"
    ]
    
    # Average activation across sequence and batch
    avg_activations = routing_weights.mean(dim=(0, 1)).cpu().numpy()
    
    return {
        expert_names[i]: float(avg_activations[i])
        for i in range(min(len(expert_names), len(avg_activations)))
    }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Demo ChartExpert-MoE")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to model configuration (if not in model_path)"
    )
    
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to chart image"
    )
    
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question about the chart"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum generation length"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature"
    )
    
    parser.add_argument(
        "--show_expert_analysis",
        action="store_true",
        help="Show expert activation analysis"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU"
    )
    
    return parser.parse_args()


def explain_reasoning_process(expert_activations, question, response):
    """Provide explanation of the reasoning process"""
    print("\n" + "="*50)
    print("REASONING PROCESS EXPLANATION")
    print("="*50)
    
    # Analyze question type
    question_lower = question.lower()
    
    print("Question Analysis:")
    print(f"📝 Question: {question}")
    
    # Determine question type
    if any(word in question_lower for word in ['trend', 'increase', 'decrease', 'change']):
        question_type = "Trend Analysis"
        expected_experts = ["trend", "geometric", "numerical"]
    elif any(word in question_lower for word in ['compare', 'higher', 'lower', 'difference']):
        question_type = "Comparison"
        expected_experts = ["geometric", "numerical", "scale"]
    elif any(word in question_lower for word in ['read', 'value', 'what is']):
        question_type = "Value Reading"
        expected_experts = ["ocr", "scale", "layout"]
    elif any(word in question_lower for word in ['how many', 'count']):
        question_type = "Counting"
        expected_experts = ["layout", "geometric", "numerical"]
    else:
        question_type = "Complex Reasoning"
        expected_experts = ["orchestrator", "integration", "query"]
    
    print(f"🎯 Question Type: {question_type}")
    print(f"🤖 Expected Key Experts: {', '.join(expected_experts)}")
    
    # Check if expected experts were activated
    print("\nExpert Utilization Check:")
    for expert in expected_experts:
        activation = expert_activations.get(expert, 0)
        if activation > 0.1:
            print(f"✅ {expert}: {activation:.3f} (as expected)")
        else:
            print(f"⚠️ {expert}: {activation:.3f} (lower than expected)")
    
    print(f"\n📄 Generated Response: {response}")


def run_mock_demo(model: ChartExpertMoE, device: torch.device):
    """Run demo with mock data"""
    print("\n" + "="*60)
    print("ChartExpert-MoE Demo - Mock Mode")
    print("="*60)
    
    # Create mock chart
    chart_image = create_mock_chart_image()
    print("\n✅ Created mock chart image (Quarterly Sales Comparison)")
    
    # Save mock chart for reference
    chart_path = "demo_chart.png"
    chart_image.save(chart_path)
    print(f"✅ Saved chart to {chart_path}")
    
    # Sample questions
    questions = [
        "What is the sales value for Q4 2023?",
        "Which quarter had the highest sales in 2022?",
        "What is the trend in sales from Q1 to Q4 in 2023?",
        "Compare the sales growth between 2022 and 2023"
    ]
    
    print("\n" + "="*60)
    print("Running inference on sample questions...")
    print("="*60)
    
    for i, question in enumerate(questions, 1):
        print(f"\n📊 Question {i}: {question}")
        print("-" * 60)
        
        # Mock inference (in practice, would use actual model)
        with torch.no_grad():
            # Create mock inputs
            batch_size = 1
            image_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
            input_ids = torch.randint(0, 1000, (batch_size, 50)).to(device)
            attention_mask = torch.ones(batch_size, 50).to(device)
            
            # Run model
            outputs = model(
                image=image_tensor,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get routing weights for expert analysis
            routing_weights = outputs.get("routing_weights", torch.randn(batch_size, 50, 10).to(device))
            
            # Analyze expert activations
            expert_activations = analyze_expert_activations(routing_weights)
            
            # Mock response based on question
            if "Q4 2023" in question:
                response = "The sales value for Q4 2023 is 62 million."
            elif "highest sales in 2022" in question:
                response = "Q4 had the highest sales in 2022 with 58 million."
            elif "trend" in question:
                response = "Sales show an upward trend from Q1 (50M) to Q4 (62M) in 2023, with steady growth each quarter."
            else:
                response = "2023 shows consistent growth compared to 2022, with an average increase of ~5M per quarter."
            
            print(f"🤖 Response: {response}")
            
            # Show expert activations
            print("\n📊 Expert Activation Analysis:")
            sorted_experts = sorted(expert_activations.items(), key=lambda x: x[1], reverse=True)
            for expert, activation in sorted_experts[:5]:  # Top 5
                bar = "█" * int(activation * 20)
                print(f"  {expert:12} {bar} {activation:.3f}")
    
    print("\n" + "="*60)
    print("Demo completed! Check 'demo_chart.png' for the generated chart.")
    print("="*60)


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Create or load model
    print("Loading ChartExpert-MoE model...")
    
    if args.model_path and os.path.exists(args.model_path):
        # Load pretrained model
        model = ChartExpertMoE.from_pretrained(args.model_path)
        print(f"✅ Loaded model from {args.model_path}")
    else:
        # Create mock model for demo
        model = create_mock_model(config)
        print("✅ Created mock model for demonstration")
    
    model = model.to(device)
    model.eval()
    
    # Run demo based on mode
    if args.image_path and os.path.exists(args.image_path):
        # Run with actual image
        print(f"\nProcessing chart image: {args.image_path}")
        
        image = Image.open(args.image_path).convert("RGB")
        
        if args.question:
            # Single question mode
            result = model.predict(
                image_path=args.image_path,
                query=args.question
            )
            
            print(f"\n📊 Question: {args.question}")
            print(f"🤖 Response: {result['response']}")
            
            if args.show_expert_analysis:
                print("\n📊 Expert Activation Analysis:")
                for expert, activation in result['expert_activations'].items():
                    bar = "█" * int(activation * 20)
                    print(f"  {expert:12} {bar} {activation:.3f}")
        else:
            # Interactive mode
            print("\nEntering interactive mode. Type 'quit' to exit.")
            while True:
                question = input("\n❓ Enter your question: ")
                if question.lower() == 'quit':
                    break
                
                result = model.predict(
                    image_path=args.image_path,
                    query=question
                )
                
                print(f"🤖 Response: {result['response']}")
    else:
        # Run mock demo
        run_mock_demo(model, device)


if __name__ == "__main__":
    main() 