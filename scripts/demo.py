#!/usr/bin/env python3
"""
Demo script for ChartExpert-MoE

This script demonstrates how to use the ChartExpert-MoE model for chart reasoning
and provides examples of different types of chart questions.
"""

import os
import sys
import argparse
import yaml
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import ChartExpertMoE
from data import ChartMuseumDataset
from transformers import AutoTokenizer


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
    
    return parser.parse_args()


def analyze_expert_activations(expert_activations, threshold=0.1):
    """Analyze and explain expert activations"""
    print("\n" + "="*50)
    print("EXPERT ACTIVATION ANALYSIS")
    print("="*50)
    
    expert_descriptions = {
        "layout": "Chart structure and element detection",
        "ocr": "Text extraction and positioning",
        "scale": "Axis scale and coordinate interpretation", 
        "geometric": "Geometric property analysis (heights, areas, etc.)",
        "trend": "Pattern and trend identification",
        "query": "Question understanding and decomposition",
        "numerical": "Mathematical reasoning and calculations",
        "integration": "Information synthesis across chart elements",
        "alignment": "Visual-textual correspondence",
        "orchestrator": "Complex multi-step reasoning coordination"
    }
    
    # Sort experts by activation strength
    sorted_experts = sorted(expert_activations.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Active Experts (threshold: {threshold}):")
    print("-" * 50)
    
    for expert, activation in sorted_experts:
        if activation > threshold:
            status = "🔥 HIGHLY ACTIVE" if activation > 0.5 else "✅ ACTIVE"
            description = expert_descriptions.get(expert, "Unknown expert")
            print(f"{status:15} {expert:12} ({activation:.3f}) - {description}")
    
    print("\nInactive Experts:")
    print("-" * 20)
    for expert, activation in sorted_experts:
        if activation <= threshold:
            description = expert_descriptions.get(expert, "Unknown expert")
            print(f"💤 {expert:12} ({activation:.3f}) - {description}")


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


def run_interactive_demo():
    """Run interactive demo with sample questions"""
    sample_questions = [
        "What is the highest value in the chart?",
        "Which category has the lowest value?", 
        "What is the trend shown in this data?",
        "Compare the values between the first and last data points.",
        "How many data points are there in total?",
        "What is the difference between the maximum and minimum values?",
        "Describe the overall pattern in this chart.",
        "What can you conclude from this visualization?"
    ]
    
    print("\n" + "="*60)
    print("CHARTEXPERT-MOE INTERACTIVE DEMO")
    print("="*60)
    print("Sample questions you can try:")
    
    for i, question in enumerate(sample_questions, 1):
        print(f"{i}. {question}")
    
    print("\nOr enter your own custom question!")
    print("Type 'quit' to exit.")
    print("-" * 60)


def main():
    """Main demo function"""
    args = parse_args()
    
    print("🚀 Loading ChartExpert-MoE...")
    
    # Load model configuration
    if args.config_path:
        config_path = args.config_path
    else:
        config_path = os.path.join(args.model_path, "config.yaml")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"⚠️ Config file not found at {config_path}, using defaults")
        config = {"llm_backbone": {"model_name": "meta-llama/Llama-2-7b-hf"}}
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["llm_backbone"]["model_name"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    try:
        model = ChartExpertMoE.from_pretrained(args.model_path, args.config_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Using randomly initialized model for demonstration...")
        model = ChartExpertMoE(config)
    
    # Set to evaluation mode
    model.eval()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"❌ Image not found: {args.image_path}")
        return
    
    # Display the chart image
    print(f"\n📊 Analyzing chart: {args.image_path}")
    
    try:
        # Load and display image
        image = Image.open(args.image_path)
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.title("Chart to Analyze")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"⚠️ Could not display image: {e}")
    
    # Run inference
    print(f"\n🤔 Question: {args.question}")
    print("🧠 Processing with ChartExpert-MoE...")
    
    try:
        with torch.no_grad():
            result = model.predict(
                image_path=args.image_path,
                query=args.question,
                max_length=args.max_length,
                temperature=args.temperature
            )
        
        response = result["response"]
        expert_activations = result["expert_activations"]
        confidence = result["confidence"]
        
        print("\n" + "="*50)
        print("CHARTEXPERT-MOE RESPONSE")
        print("="*50)
        print(f"💬 Answer: {response}")
        print(f"🎯 Confidence: {confidence:.3f}")
        
        if args.show_expert_analysis:
            analyze_expert_activations(expert_activations)
            explain_reasoning_process(expert_activations, args.question, response)
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        print("This might be due to missing model weights or configuration issues.")
    
    # Show sample questions for future use
    print("\n" + "="*50)
    print("TRY MORE QUESTIONS")
    print("="*50)
    print("Here are some sample questions you can try with different charts:")
    
    sample_questions = [
        "What is the trend in the data over time?",
        "Which category has the highest value?",
        "What is the relationship between X and Y variables?", 
        "How much did the value change between 2020 and 2021?",
        "What percentage of the total does each segment represent?",
        "Are there any outliers in the data?",
        "What pattern can you observe in this chart?",
        "What conclusions can be drawn from this visualization?"
    ]
    
    for i, question in enumerate(sample_questions, 1):
        print(f"{i}. {question}")
    
    print(f"\nTo try another question, run:")
    print(f"python {sys.argv[0]} --model_path {args.model_path} --image_path <new_image> --question '<new_question>'")


if __name__ == "__main__":
    main() 