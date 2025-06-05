#!/usr/bin/env python3
"""Test real model availability"""

import torch
from transformers import AutoTokenizer, AutoModel

print('🔧 Testing real model availability...')

models_to_test = [
    "microsoft/DialoGPT-medium",
    "microsoft/DialoGPT-small", 
    "distilbert-base-uncased",
    "bert-base-uncased"
]

available_models = []

for model_name in models_to_test:
    try:
        print(f"\n🔍 Testing {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        params = sum(p.numel() for p in model.parameters())
        hidden_size = getattr(model.config, 'hidden_size', getattr(model.config, 'd_model', 'unknown'))
        
        print(f'✅ {model_name}: Available')
        print(f'   Vocab size: {tokenizer.vocab_size:,}')
        print(f'   Hidden size: {hidden_size}')
        print(f'   Parameters: {params:,}')
        
        available_models.append({
            'name': model_name,
            'vocab_size': tokenizer.vocab_size,
            'hidden_size': hidden_size,
            'parameters': params
        })
        
    except Exception as e:
        print(f'❌ {model_name}: {e}')

print(f"\n🎯 Summary: {len(available_models)}/{len(models_to_test)} models available")
print("\n📊 Available models for MoE integration:")
for model in available_models:
    print(f"   • {model['name']}: {model['parameters']:,} params, {model['hidden_size']} hidden")

print("\n🚀 Ready for real model MoE training!") 