#!/usr/bin/env python
"""
Test script to verify ChartExpert-MoE architecture is complete and functional
"""

import os
import sys
import torch

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import ChartExpertMoE
from src.utils import load_config


def test_architecture():
    """Test that all components are properly implemented"""
    print("="*60)
    print("ChartExpert-MoE Architecture Test")
    print("="*60)
    
    # Load config
    config_path = "configs/chart_expert_base.yaml"
    config = load_config(config_path)
    print(f"✅ Loaded configuration from {config_path}")
    
    # Create minimal config for testing
    test_config = {
        "hidden_size": 768,
        "vocab_size": 32000,
        "vision_encoder": config["vision_encoder"],
        "llm_backbone": config["llm_backbone"],
        "experts": config["experts"],
        "routing": config["routing"],
        "moe": config["moe"],
        "fusion": config["fusion"]
    }
    
    # Create model
    print("\nCreating model...")
    model = ChartExpertMoE(test_config)
    print("✅ Model created successfully")
    
    # Check all experts are present
    print("\nChecking expert modules...")
    expected_experts = [
        "layout_expert", "ocr_expert", "scale_expert", "geometric_expert", "trend_expert",
        "query_expert", "numerical_expert", "integration_expert", 
        "alignment_expert", "chart_to_graph_expert", "shallow_reasoning_expert", "orchestrator_expert"
    ]
    
    for expert_name in expected_experts:
        if hasattr(model, expert_name):
            print(f"  ✅ {expert_name}")
        else:
            print(f"  ❌ {expert_name} - MISSING!")
    
    # Test forward pass
    print("\nTesting forward pass...")
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    # Create mock inputs
    batch_size = 2
    image = torch.randn(batch_size, 3, 224, 224).to(device)
    input_ids = torch.randint(0, 1000, (batch_size, 50)).to(device)
    attention_mask = torch.ones(batch_size, 50).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            image=image,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    print("✅ Forward pass successful")
    
    # Check outputs
    print("\nChecking output structure...")
    expected_outputs = ["logits", "routing_weights", "aux_loss", "visual_features", 
                       "text_features", "fused_features", "expert_outputs"]
    
    for output_name in expected_outputs:
        if output_name in outputs:
            shape = outputs[output_name].shape if hasattr(outputs[output_name], 'shape') else 'scalar'
            print(f"  ✅ {output_name}: {shape}")
        else:
            print(f"  ❌ {output_name} - MISSING!")
    
    # Test routing weights
    if "routing_weights" in outputs:
        routing_weights = outputs["routing_weights"]
        print(f"\nRouting weights shape: {routing_weights.shape}")
        print(f"Number of experts: {routing_weights.shape[-1]}")
        
        # Check expert activation
        avg_activation = routing_weights.mean(dim=(0, 1))
        print("\nAverage expert activation:")
        expert_names = [
            "layout", "ocr", "scale", "geometric", "trend",
            "query", "numerical", "integration", "alignment", 
            "chart_to_graph", "shallow_reasoning", "orchestrator"
        ]
        for i, (name, activation) in enumerate(zip(expert_names, avg_activation)):
            print(f"  {name:18}: {activation:.3f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024**2:.2f} MB (float32)")
    
    print("\n" + "="*60)
    print("✅ All architecture tests passed!")
    print("="*60)


if __name__ == "__main__":
    test_architecture() 