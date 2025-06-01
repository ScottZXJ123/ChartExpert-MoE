# ChartExpert-MoE Quick Start Guide

## 🚀 Getting Started

This guide will help you quickly get started with ChartExpert-MoE, a state-of-the-art Mixture-of-Experts Vision-Language Model for complex chart reasoning.

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/ChartExpert-MoE.git
cd ChartExpert-MoE
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -e .
```

## Quick Usage

### 1. Interactive Demo

Try the model with the interactive demo:

```bash
python scripts/demo.py
```

This will:
- Load a pre-configured model
- Allow you to upload chart images and ask questions
- Show expert activation patterns and reasoning steps

### 2. Training from Scratch

To train ChartExpert-MoE on your data:

```bash
python scripts/train.py --config configs/chart_expert_base.yaml
```

For distributed training:
```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/chart_expert_base.yaml
```

### 3. Using Pre-trained Models

```python
from src.models import ChartExpertMoE

# Load model from checkpoint
model = ChartExpertMoE.from_pretrained("path/to/checkpoint")

# Make predictions
result = model.predict(
    image_path="chart.png",
    query="What is the trend in Q3 sales?"
)

print(result["response"])
print("Expert activations:", result["expert_activations"])
```

### 4. Custom Dataset Integration

```python
from src.data.datasets import ChartDataset

# Create custom dataset
dataset = ChartDataset(
    data_path="path/to/your/data",
    tokenizer=tokenizer,
    transform=transform
)

# Use with DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
```

## Key Features

### Expert Modules
- **Visual-Spatial Experts**: Layout detection, OCR, scale interpretation, geometric analysis, trend detection
- **Semantic Experts**: Query understanding, numerical reasoning, knowledge integration
- **Cross-Modal Experts**: Visual-textual alignment, chart-to-graph transformation
- **Cognitive Experts**: Shallow/deep reasoning orchestration

### Training Stages
1. **Foundation**: Basic multimodal pre-training
2. **Joint Pre-training**: Vision-language alignment
3. **Chart Tuning**: Chart-specific understanding
4. **Expert Specialization**: Fine-tune individual experts
5. **ChartMuseum Fine-tuning**: Optimize for benchmark

### Configuration

Modify `configs/chart_expert_base.yaml` to customize:
- Model architecture (vision encoder, LLM backbone)
- Expert configurations
- Training hyperparameters
- Data processing settings

## Common Tasks

### Evaluate on ChartMuseum
```bash
python scripts/evaluate.py --dataset chartmuseum --checkpoint path/to/model
```

### Export Model
```python
model.save_pretrained("path/to/save")
```

### Analyze Expert Usage
```python
# During inference
result = model.predict(image_path, query)
expert_usage = result["expert_activations"]

# Visualize which experts were most active
import matplotlib.pyplot as plt
plt.bar(expert_usage.keys(), expert_usage.values())
plt.xlabel("Expert")
plt.ylabel("Activation")
plt.xticks(rotation=45)
plt.show()
```

## Troubleshooting

### Out of Memory
- Reduce batch size in config
- Enable gradient checkpointing
- Use mixed precision training

### Slow Training
- Enable distributed training
- Use gradient accumulation
- Optimize data loading with more workers

### Poor Performance
- Check if using correct training stage
- Verify data preprocessing
- Adjust learning rate and warmup

## Next Steps

- Read the [full documentation](README.md)
- Explore [example notebooks](examples/)
- Check [architecture details](ARCHITECTURE_IMPLEMENTATION_STATUS.md)
- Join our community discussions

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/your-org/ChartExpert-MoE/issues)
- Documentation: [Full docs](docs/)
- Research Paper: [arXiv link]

Happy chart reasoning! 📊🤖 