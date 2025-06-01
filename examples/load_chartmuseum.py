"""
Example script demonstrating how to load and explore the ChartMuseum dataset

This script shows:
1. How to load the ChartMuseum dataset
2. Basic dataset exploration
3. Visualization of sample data
4. Preparing data for training/evaluation
"""

import os
import sys
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from datasets import load_dataset
from transformers import AutoTokenizer
from data import ChartMuseumDataset
import matplotlib.pyplot as plt
from PIL import Image
import torch


def explore_chartmuseum_dataset():
    """Explore the ChartMuseum dataset structure and content"""
    
    print("🚀 Loading ChartMuseum Dataset...")
    print("=" * 50)
    
    # Load the dataset directly using datasets library
    try:
        dataset = load_dataset("lytang/ChartMuseum")
        print("✅ Dataset loaded successfully!")
        print(f"📊 Available splits: {list(dataset.keys())}")
        
        # Get test split for exploration
        test_data = dataset["test"]
        print(f"📈 Test set size: {len(test_data)}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Make sure you're connected to the internet and have access to Hugging Face datasets.")
        return
    
    # Explore dataset structure
    print("\n📋 Dataset Schema:")
    print("-" * 30)
    if len(test_data) > 0:
        sample = test_data[0]
        for key, value in sample.items():
            if isinstance(value, Image.Image):
                print(f"  {key}: PIL Image ({value.size})")
            elif isinstance(value, str):
                print(f"  {key}: String (length: {len(value)})")
            else:
                print(f"  {key}: {type(value)} - {value}")
    
    # Analyze dataset statistics
    print("\n📊 Dataset Statistics:")
    print("-" * 30)
    
    # Chart types distribution
    chart_types = {}
    reasoning_types = {}
    
    for item in test_data:
        chart_type = item.get("chart_type", "unknown")
        reasoning_type = item.get("reasoning_type", "unknown")
        
        chart_types[chart_type] = chart_types.get(chart_type, 0) + 1
        reasoning_types[reasoning_type] = reasoning_types.get(reasoning_type, 0) + 1
    
    print("Chart Types:")
    for chart_type, count in sorted(chart_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(test_data)) * 100
        print(f"  {chart_type}: {count} ({percentage:.1f}%)")
    
    print("\nReasoning Types:")
    for reasoning_type, count in sorted(reasoning_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(test_data)) * 100
        print(f"  {reasoning_type}: {count} ({percentage:.1f}%)")
    
    return test_data


def visualize_samples(dataset, num_samples=4):
    """Visualize sample data from the dataset"""
    
    print(f"\n🎨 Visualizing {num_samples} sample charts...")
    print("=" * 50)
    
    # Create subplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        # Display image
        if sample.get("image"):
            axes[i].imshow(sample["image"])
            axes[i].axis('off')
            
            # Create title with information
            chart_type = sample.get("chart_type", "Unknown")
            reasoning_type = sample.get("reasoning_type", "Unknown") 
            question = sample.get("question", "")[:50] + "..." if len(sample.get("question", "")) > 50 else sample.get("question", "")
            
            title = f"Chart: {chart_type}\nReasoning: {reasoning_type}\nQ: {question}"
            axes[i].set_title(title, fontsize=10, pad=10)
        else:
            axes[i].text(0.5, 0.5, "No Image", ha='center', va='center')
            axes[i].set_xlim(0, 1)
            axes[i].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.suptitle("ChartMuseum Dataset Samples", fontsize=16, y=1.02)
    plt.show()
    
    # Print detailed information for first sample
    if len(dataset) > 0:
        sample = dataset[0]
        print("\n📝 Detailed Sample Analysis:")
        print("-" * 40)
        print(f"Question: {sample.get('question', 'N/A')}")
        print(f"Answer: {sample.get('answer', 'N/A')}")
        print(f"Chart Type: {sample.get('chart_type', 'N/A')}")
        print(f"Reasoning Type: {sample.get('reasoning_type', 'N/A')}")


def test_dataset_loader():
    """Test the custom ChartMuseumDataset loader"""
    
    print("\n🧪 Testing Custom Dataset Loader...")
    print("=" * 50)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset instance
    try:
        dataset = ChartMuseumDataset(
            tokenizer=tokenizer,
            split="test",
            max_length=512,
            image_size=(224, 224)
        )
        
        print(f"✅ Custom dataset loader working!")
        print(f"📊 Dataset size: {len(dataset)}")
        
        # Test data loading
        sample = dataset[0]
        print(f"📝 Sample data shapes:")
        print(f"  Image: {sample['image'].shape}")
        print(f"  Input IDs: {sample['input_ids'].shape}")
        print(f"  Attention Mask: {sample['attention_mask'].shape}")
        print(f"  Labels: {sample['labels'].shape}")
        
        # Test filtering
        reasoning_types = dataset.get_reasoning_types()
        print(f"🏷️ Available reasoning types: {reasoning_types}")
        
        if reasoning_types:
            filtered_dataset = dataset.filter_by_reasoning_type(reasoning_types[0])
            print(f"🔍 Filtered dataset size for '{reasoning_types[0]}': {len(filtered_dataset)}")
        
    except Exception as e:
        print(f"❌ Error testing custom dataset loader: {e}")


def analyze_complexity():
    """Analyze the complexity of questions in ChartMuseum"""
    
    print("\n🧠 Analyzing Question Complexity...")
    print("=" * 50)
    
    try:
        dataset = load_dataset("lytang/ChartMuseum")["test"]
        
        # Question length analysis
        question_lengths = []
        answer_lengths = []
        
        for item in dataset:
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            question_lengths.append(len(question.split()))
            answer_lengths.append(len(str(answer).split()))
        
        print(f"📏 Question Statistics:")
        print(f"  Average length: {sum(question_lengths) / len(question_lengths):.1f} words")
        print(f"  Max length: {max(question_lengths)} words")
        print(f"  Min length: {min(question_lengths)} words")
        
        print(f"\n📏 Answer Statistics:")
        print(f"  Average length: {sum(answer_lengths) / len(answer_lengths):.1f} words")
        print(f"  Max length: {max(answer_lengths)} words")  
        print(f"  Min length: {min(answer_lengths)} words")
        
        # Complexity keywords analysis
        complexity_keywords = {
            "comparison": ["compare", "higher", "lower", "greater", "less", "difference"],
            "calculation": ["calculate", "sum", "total", "average", "percentage"],
            "trend": ["trend", "increase", "decrease", "change", "over time"],
            "visual": ["color", "pattern", "shape", "position", "size"],
            "reasoning": ["why", "because", "reason", "conclude", "infer"]
        }
        
        keyword_counts = {category: 0 for category in complexity_keywords}
        
        for item in dataset:
            question = item.get("question", "").lower()
            for category, keywords in complexity_keywords.items():
                if any(keyword in question for keyword in keywords):
                    keyword_counts[category] += 1
        
        print(f"\n🔍 Question Categories:")
        total_samples = len(dataset)
        for category, count in keyword_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  {category.capitalize()}: {count} ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"❌ Error analyzing complexity: {e}")


def main():
    """Main function to run all examples"""
    
    print("🎯 ChartMuseum Dataset Explorer")
    print("=" * 60)
    print("This script demonstrates how to load and explore the ChartMuseum dataset")
    print("for the ChartExpert-MoE project.\n")
    
    # 1. Explore dataset
    dataset = explore_chartmuseum_dataset()
    
    if dataset is not None:
        # 2. Visualize samples
        visualize_samples(dataset)
        
        # 3. Test custom loader
        test_dataset_loader()
        
        # 4. Analyze complexity
        analyze_complexity()
    
    print("\n✅ Dataset exploration completed!")
    print("\n📚 Next Steps:")
    print("1. Use the ChartMuseumDataset class in your training scripts")
    print("2. Experiment with different reasoning types and chart types")
    print("3. Train your ChartExpert-MoE model on this challenging dataset")
    print("4. Evaluate performance on different complexity levels")


if __name__ == "__main__":
    main() 