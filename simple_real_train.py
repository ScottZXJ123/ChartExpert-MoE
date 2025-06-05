#!/usr/bin/env python3
import sys
import os
sys.path.append('./src')

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as transforms
import wandb

# Direct imports that work
from models.chart_expert_moe import ChartExpertMoE

class SimpleRealDataset:
    def __init__(self, tokenizer, max_samples=50):
        self.tokenizer = tokenizer
        print('Loading ChartMuseum...')
        ds = load_dataset('lytang/ChartMuseum', split='test')
        self.data = list(ds.select(range(min(max_samples, len(ds)))))
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print(f'Loaded {len(self.data)} real samples')
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        
        # Process image
        if isinstance(item['image'], str):
            try:
                image = Image.open(item['image']).convert('RGB')
            except:
                image = Image.new('RGB', (224, 224), 'white')
        else:
            image = item['image'].convert('RGB')
        
        processed_image = self.transform(image)
        
        # Tokenize
        prompt = f'Question: {question}\nAnswer:'
        inputs = self.tokenizer(prompt, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        
        answer_inputs = self.tokenizer(str(answer), return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        labels = answer_inputs['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'image': processed_image,
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels
        }

def main():
    # Configuration
    config = {
        'hidden_size': 768, 'vocab_size': 50257, 'num_heads': 12,
        'use_hierarchical_experts': True, 'use_flash_attention': False,
        'simple_confidence_threshold': 0.95, 'medium_confidence_threshold': 0.90,
        'complex_confidence_threshold': 0.85, 'num_early_exit_layers': 3,
        'kv_cache_size_limit': 1073741824, 'min_experts': 1, 'aux_loss_weight': 0.01,
        'vision_encoder': {'encoder_type': 'mock', 'hidden_size': 768},
        'llm_backbone': {'model_name': 'microsoft/DialoGPT-medium', 'hidden_size': 1024, 'vocab_size': 50257, 'use_mock': True},
        'fusion': {'hidden_size': 768, 'num_heads': 12, 'dropout': 0.1},
        'routing': {'hidden_size': 768, 'num_experts': 12, 'top_k': 2, 'dropout': 0.1},
        'moe': {'hidden_size': 768, 'num_experts': 12, 'top_k': 2, 'capacity_factor': 1.25, 'dropout': 0.1},
        'experts': {k: {'hidden_size': 768} for k in ['layout', 'ocr', 'scale', 'geometric', 'trend', 'query', 'numerical', 'integration', 'alignment', 'chart_to_graph', 'shallow_reasoning', 'orchestrator']}
    }

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize wandb
    wandb.init(project='chartexpert-real-test', config={'samples': 20, 'epochs': 1})

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = ChartExpertMoE(config).to(device)
    print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters')

    # Dataset and loader
    dataset = SimpleRealDataset(tokenizer, max_samples=20)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Training loop
    model.train()
    for epoch in range(1):
        total_loss = 0
        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(
                image=batch['image'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs['loss']
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            print(f'Batch {batch_idx + 1}, Loss: {loss.item():.4f}')
            
            wandb.log({'batch_loss': loss.item(), 'batch': batch_idx})
            
            if batch_idx >= 4:  # Just a few batches for testing
                break

    avg_loss = total_loss / (batch_idx + 1)
    print(f'✅ Training completed! Average loss: {avg_loss:.4f}')

    wandb.log({'final_avg_loss': avg_loss})
    wandb.finish()

    print('🎉 Real data training test successful!')

if __name__ == '__main__':
    main() 