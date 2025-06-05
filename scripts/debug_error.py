#!/usr/bin/env python3
"""
Debug script to isolate the visual_features error
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.chart_expert_moe import ChartExpertMoE

def test_basic_model():
    """Test basic model functionality"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Minimal configuration
    config = {
        "hidden_size": 768,
        "vocab_size": 32000,
        "num_heads": 12,
        "use_hierarchical_experts": True,
        "use_flash_attention": True,
        "simple_confidence_threshold": 0.95,
        "medium_confidence_threshold": 0.90,
        "complex_confidence_threshold": 0.85,
        "num_early_exit_layers": 3,
        "kv_cache_size_limit": 1024 * 1024 * 1024,
        "vision_encoder": {
            "encoder_type": "mock",
            "hidden_size": 768,
            "model_name": "openai/clip-vit-base-patch32",
            "use_native_resolution": False,
            "use_2d_rope": False
        },
        "llm_backbone": {
            "model_name": "microsoft/DialoGPT-medium",
            "hidden_size": 4096,
            "use_mock": True
        },
        "fusion": {
            "hidden_size": 768, 
            "num_heads": 12,
            "fusion_type": "attention"
        },
        "routing": {
            "hidden_size": 768, 
            "num_experts": 12,
            "top_k": 2
        },
        "moe": {
            "hidden_size": 768,
            "num_experts": 12
        },
        "experts": {
            "layout": {"hidden_size": 768},
            "ocr": {"hidden_size": 768},
            "scale": {"hidden_size": 768},
            "geometric": {"hidden_size": 768},
            "trend": {"hidden_size": 768},
            "query": {"hidden_size": 768},
            "numerical": {"hidden_size": 768},
            "integration": {"hidden_size": 768},
            "alignment": {"hidden_size": 768},
            "chart_to_graph": {"hidden_size": 768},
            "shallow_reasoning": {"hidden_size": 768},
            "orchestrator": {"hidden_size": 768}
        }
    }
    
    print("Creating model...")
    try:
        model = ChartExpertMoE(config)
        model = model.to(device)
        model.eval()
        print("✅ Model created successfully")
        
        # Test inputs
        image = torch.randn(2, 3, 224, 224, device=device)
        input_ids = torch.randint(0, config["vocab_size"], (2, 64), device=device)
        attention_mask = torch.ones(2, 64, device=device)
        
        print("Testing forward pass...")
        try:
            outputs = model.forward(image, input_ids, attention_mask)
            print("✅ Forward pass successful")
            print(f"Output keys: {list(outputs.keys())}")
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("Testing smart_inference...")
        try:
            smart_outputs = model.smart_inference(
                image, input_ids, attention_mask,
                use_early_exit=True,
                use_kv_cache=True
            )
            print("✅ Smart inference successful")
            print(f"Smart output keys: {list(smart_outputs.keys())}")
        except Exception as e:
            print(f"❌ Smart inference failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_model() 