# ChartExpert-MoE: A Novel MoE-VLM Architecture for Complex Chart Reasoning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

ChartExpert-MoE is a specialized Mixture-of-Experts Vision-Language Model designed to address the "visual reasoning gap" in chart understanding. Unlike existing VLMs that often rely heavily on textual cues, our architecture employs specialized expert modules to handle different aspects of chart comprehension, from visual-spatial analysis to complex multi-step reasoning.

### Key Innovation
- **🎯 Targets ChartMuseum benchmark**: Addressing the 93% human vs 38.5-63% model accuracy gap
- **🧠 12 Specialized Expert Modules**: Fine-grained experts for every aspect of chart reasoning
- **🔄 Dynamic Routing**: Intelligent content-aware, modality-aware, and context-sensitive routing
- **🔗 Advanced Fusion**: Multi-strategy fusion including dynamic gated and graph-based approaches
- **📈 Multi-stage Training**: Progressive training from foundation to chart specialization

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ScottZXJ123/ChartExpert-MoE.git
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

### Usage

#### Interactive Demo
Try the model with the interactive demo:
```bash
python scripts/demo.py
```

#### Python API
```python
from src.models.chart_expert_moe import ChartExpertMoE

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

#### Training from Scratch
```bash
# Basic training
python scripts/train.py --config configs/chart_expert_base.yaml

# Distributed training
torchrun --nproc_per_node=4 scripts/train.py --config configs/chart_expert_base.yaml
```

## 🏗️ Architecture

### 🎯 Expert Modules (12 Specialized Experts)

#### Visual-Spatial & Structural Experts (5 experts)
- **Layout Detection Expert**: Object detection, spatial relationship encoding, element classification
- **OCR & Text Grounding Expert**: High-precision text extraction, position encoding, text-visual alignment
- **Scale & Coordinate Interpretation Expert**: Scale type classification, coordinate mapping, numerical processing
- **Geometric Property Expert**: Shape analysis, geometric feature extraction, comparison operations
- **Trend & Pattern Perception Expert**: LSTM-based sequence modeling, pattern detection, trend classification

#### Semantic & Relational Experts (3 experts)
- **Query Deconstruction Expert**: Intent classification, complexity estimation, query decomposition
- **Numerical & Logical Reasoning Expert**: Arithmetic/comparison/aggregation processors, reasoning chains
- **Knowledge Integration Expert**: Multi-source attention, information fusion, consistency checking

#### Cross-Modal Fusion & Reasoning Experts (2 experts)
- **Visual-Textual Alignment Expert**: Bidirectional attention, fine-grained alignment, confidence estimation
- **Chart-to-Graph Transformation Expert**: Graph neural networks (GAT), node/edge identification, structure consistency

#### Cognitive Effort Modulation Experts (2 experts)
- **Shallow Reasoning Expert**: Fast processing, simple pattern matching (512 hidden units for efficiency)
- **Deep Reasoning Orchestrator Expert**: Multi-step reasoning, expert orchestration (2048 hidden units)

### 🔄 Dynamic Routing System

- **Content-Aware Routing**: Routes based on visual content complexity
- **Modality-Aware Routing**: Handles text, visual, and multimodal inputs differently
- **Context-Sensitive Routing**: Considers conversation history and task context
- **Load Balancing**: Ensures even expert utilization with auxiliary losses
- **Batch Priority Routing (BPR)**: Prioritizes important tokens

### 🔗 Advanced Fusion Strategies

- **Dynamic Gated Fusion**: Learnable gating with adaptive control
- **FiLM-Guided Fusion**: Language-guided visual modulation
- **Graph-Based Fusion**: GAT-based chart-to-graph processing
- **Structural Chart Fusion**: Spatial hierarchy modeling

### 🏛️ Base Models Support

#### Vision Encoders
- **MoonViT**: Native resolution processing, variable patch sizes
- **DINOv2**: Self-supervised robust features
- **CLIP/SigLIP**: Multimodal pre-training
- **SAM**: Edge-aware features for chart elements

#### LLM Backbones
- **Llama 3 series**: Advanced reasoning capabilities
- **Qwen2.5-VL**: Multimodal understanding
- **Gemma 3**: Efficient processing

## 📊 Training Pipeline

### 5-Stage Training Process

1. **Foundation (Stage 1)**: Basic multimodal pre-training
2. **Joint Pre-training (Stage 2)**: Vision-language alignment
3. **Chart Tuning (Stage 3)**: Chart-specific understanding
4. **Expert Specialization (Stage 4)**: Fine-tune individual experts
5. **ChartMuseum Fine-tuning (Stage 5)**: Optimize for benchmark

```bash
# Run all stages sequentially
python scripts/train.py --stage foundation --config configs/stage1_foundation.yaml
python scripts/train.py --stage joint_pretrain --config configs/stage2_joint.yaml
python scripts/train.py --stage chart_tuning --config configs/stage3_chart.yaml
python scripts/train.py --stage expert_specialization --config configs/stage4_experts.yaml
python scripts/train.py --stage chartmuseum_finetune --config configs/stage5_chartmuseum.yaml
```

## 📈 Dataset Support

### ChartMuseum Integration
```python
from src.data.datasets import ChartMuseumDataset

# Load ChartMuseum with native support
dataset = ChartMuseumDataset()
# Supports reasoning type filtering, chart type filtering
```

### Additional Datasets
- **ChartQA**: Chart question answering
- **PlotQA**: Plot-based visual reasoning
- **Custom Datasets**: Easy integration with `ChartDataset` class

## 🔧 Evaluation

```bash
# Evaluate on ChartMuseum
python scripts/evaluate.py --dataset chartmuseum --checkpoint path/to/model

# Run comprehensive benchmarks
python scripts/benchmark.py --config configs/evaluation.yaml
```

### Metrics
- **ChartMuseum Accuracy**: Primary benchmark metric
- **Visual Reasoning Gap**: Human vs model performance analysis
- **Expert Activation Analysis**: Understanding expert usage patterns
- **Error Categorization**: Detailed failure analysis

## ⚙️ Advanced Features

### Performance Optimization
- **FlashAttention**: Optimized attention computation
- **Quantization**: INT8 dynamic/static quantization
- **Pruning**: Structured and expert-specific pruning
- **Knowledge Distillation**: Teacher-student framework
- **Mixed Precision**: Automatic mixed precision training

### Configuration
Customize everything via `configs/chart_expert_base.yaml`:
- Model architecture (vision encoder, LLM backbone)
- Expert configurations and routing strategies
- Training hyperparameters and data settings
- Optimization and performance settings

## 📊 Performance

### ChartMuseum Benchmark Results
- **Baseline VLMs**: 38.5-63% accuracy
- **Human Performance**: ~93% accuracy
- **ChartExpert-MoE**: [Results pending training completion]

### Expert Usage Analysis
```python
# Analyze which experts are most active
result = model.predict(image_path, query)
expert_usage = result["expert_activations"]

import matplotlib.pyplot as plt
plt.bar(expert_usage.keys(), expert_usage.values())
plt.xticks(rotation=45)
plt.show()
```

## 🛠️ Implementation Status

### ✅ Fully Implemented Components
- **✅ Expert Modules**: All 12 experts with specialized architectures
- **✅ Dynamic Routing**: Content/modality/context-aware routing with load balancing
- **✅ Base Models**: Complete vision encoder + LLM support with 2D RoPE
- **✅ Fusion Strategies**: All fusion approaches including FiLM and graph-based
- **✅ MoE Architecture**: Complete with optimizations (top-k gating, BPR)
- **✅ Training Pipeline**: Full 5-stage training with distributed support
- **✅ Data Handling**: ChartMuseum integration + additional datasets
- **✅ Evaluation Framework**: Comprehensive metrics and benchmarking
- **✅ Performance Optimization**: FlashAttention, quantization, pruning

### Repository Structure
```
ChartExpert-MoE/
├── configs/                          # Configuration files
├── src/
│   ├── data/                        # Data handling and datasets
│   ├── evaluation/                  # Evaluation modules
│   ├── experts/                     # 12 specialized expert modules
│   ├── fusion/                      # Multi-strategy fusion
│   ├── models/                      # Core MoE model architecture
│   ├── routing/                     # Dynamic routing system
│   ├── training/                    # Training infrastructure
│   └── utils/                       # Utility functions
├── scripts/                          # Training, demo, and evaluation scripts
├── examples/                         # Usage examples
└── checkpoints/                      # Model checkpoints
```

## 🔍 Troubleshooting

### Common Issues

**Out of Memory**
- Reduce batch size in config
- Enable gradient checkpointing
- Use mixed precision training

**Slow Training**
- Enable distributed training with `torchrun`
- Use gradient accumulation
- Optimize data loading with more workers

**Poor Performance**
- Verify correct training stage
- Check data preprocessing
- Adjust learning rate and warmup schedules

## 📝 Citation

```bibtex
@article{chartexpert2024,
  title={ChartExpert-MoE: A Novel MoE-VLM Architecture for Complex Chart Reasoning},
  author={ChartExpert Team},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## 📄 License

MIT License

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 💬 Support

- **GitHub Issues**: [Create an issue](https://github.com/ScottZXJ123/ChartExpert-MoE/issues)
- **Research Paper**: [arXiv link] (coming soon)
- **Documentation**: This README and inline code documentation

---

**Built with ❤️ for advancing chart understanding in AI** 📊🤖 
Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project. 