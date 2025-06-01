# ChartExpert-MoE Project Structure

This document provides an overview of the ChartExpert-MoE repository structure and explains the purpose of each component.

## 📁 Directory Structure

```
ChartExpert-MoE/
│
├── 📄 README.md                    # Main project documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package installation script
├── 📄 .gitignore                   # Git ignore patterns
├── 📄 PROJECT_STRUCTURE.md         # This file
│
├── 📁 src/                         # Main source code
│   ├── 📄 __init__.py
│   │
│   ├── 📁 models/                  # Core model architectures
│   │   ├── 📄 __init__.py
│   │   ├── 📄 chart_expert_moe.py  # Main ChartExpert-MoE model
│   │   ├── 📄 base_models.py       # Vision encoder & LLM backbone
│   │   └── 📄 moe_layer.py         # MoE implementation with routing
│   │
│   ├── 📁 experts/                 # Specialized expert modules
│   │   ├── 📄 __init__.py
│   │   ├── 📄 base_expert.py       # Base expert class
│   │   ├── 📄 visual_spatial.py    # Visual-spatial experts
│   │   ├── 📄 semantic_relational.py   # Semantic experts (to be created)
│   │   ├── 📄 cross_modal.py       # Cross-modal experts (to be created)
│   │   └── 📄 cognitive_modulation.py  # Cognitive experts (to be created)
│   │
│   ├── 📁 routing/                 # Expert routing mechanisms
│   │   ├── 📄 __init__.py
│   │   ├── 📄 dynamic_router.py    # Dynamic content-aware routing
│   │   ├── 📄 hierarchical_router.py   # Hierarchical routing (to be created)
│   │   ├── 📄 skill_based_router.py    # Skill-based routing (to be created)
│   │   └── 📄 rl_router.py         # RL-enhanced routing (to be created)
│   │
│   ├── 📁 fusion/                  # Multimodal fusion strategies
│   │   ├── 📄 __init__.py          # (to be created)
│   │   ├── 📄 dynamic_fusion.py    # Dynamic gated fusion (to be created)
│   │   └── 📄 attention_fusion.py  # Attention-based fusion (to be created)
│   │
│   ├── 📁 data/                    # Data loading and preprocessing
│   │   ├── 📄 __init__.py
│   │   ├── 📄 datasets.py          # Dataset classes (ChartMuseum, etc.)
│   │   ├── 📄 preprocessing.py     # Data preprocessing (to be created)
│   │   └── 📄 data_loader.py       # Data loading utilities (to be created)
│   │
│   ├── 📁 training/                # Training utilities
│   │   ├── 📄 __init__.py          # (to be created)
│   │   ├── 📄 trainer.py           # Training logic (to be created)
│   │   └── 📄 multi_stage_trainer.py   # Multi-stage training (to be created)
│   │
│   ├── 📁 evaluation/              # Evaluation metrics and utilities
│   │   ├── 📄 __init__.py          # (to be created)
│   │   ├── 📄 chart_evaluator.py   # Chart-specific evaluation (to be created)
│   │   └── 📄 metrics.py           # Evaluation metrics (to be created)
│   │
│   └── 📁 utils/                   # Utility functions
│       ├── 📄 __init__.py          # (to be created)
│       ├── 📄 logging.py           # Logging utilities (to be created)
│       └── 📄 config.py            # Configuration utilities (to be created)
│
├── 📁 configs/                     # Configuration files
│   ├── 📄 chart_expert_base.yaml   # Base model configuration
│   ├── 📄 stage1_foundation.yaml   # Stage 1 training config (to be created)
│   ├── 📄 stage2_joint.yaml        # Stage 2 training config (to be created)
│   ├── 📄 stage3_chart.yaml        # Stage 3 training config (to be created)
│   ├── 📄 stage4_experts.yaml      # Stage 4 training config (to be created)
│   └── 📄 stage5_chartmuseum.yaml  # Stage 5 training config (to be created)
│
├── 📁 scripts/                     # Executable scripts
│   ├── 📄 train.py                 # Main training script
│   ├── 📄 demo.py                  # Demo/inference script
│   ├── 📄 evaluate.py              # Evaluation script (to be created)
│   └── 📄 ablation.py              # Ablation studies (to be created)
│
├── 📁 examples/                    # Example usage scripts
│   ├── 📄 load_chartmuseum.py      # ChartMuseum dataset example
│   ├── 📄 basic_training.py        # Basic training example (to be created)
│   └── 📄 inference_example.py     # Inference example (to be created)
│
├── 📁 experiments/                 # Experimental results
│   ├── 📄 README.md                # Experiments documentation (to be created)
│   └── 📁 results/                 # Results storage (to be created)
│
├── 📁 docs/                        # Documentation
│   ├── 📄 ARCHITECTURE.md          # Architecture documentation (to be created)
│   ├── 📄 TRAINING.md              # Training guide (to be created)
│   ├── 📄 API.md                   # API documentation (to be created)
│   └── 📄 CONTRIBUTING.md          # Contribution guidelines (to be created)
│
├── 📁 tests/                       # Unit tests
│   ├── 📄 __init__.py              # (to be created)
│   ├── 📄 test_models.py           # Model tests (to be created)
│   ├── 📄 test_experts.py          # Expert tests (to be created)
│   ├── 📄 test_routing.py          # Routing tests (to be created)
│   └── 📄 test_data.py             # Data tests (to be created)
│
├── 📁 checkpoints/                 # Model checkpoints (created during training)
├── 📁 logs/                        # Training logs (created during training)
└── 📁 data/                        # Data storage
    ├── 📁 raw/                     # Raw datasets
    ├── 📁 processed/               # Processed datasets
    └── 📁 cache/                   # Dataset cache
```

## 🏗️ Architecture Components

### 1. **Core Models** (`src/models/`)
- **ChartExpertMoE**: Main model class integrating all components
- **VisionEncoder**: Supports multiple vision encoders (CLIP, SigLIP, DINOv2, MoonViT)
- **LLMBackbone**: Language model integration (Llama, Qwen, Gemma)
- **MoELayer**: Mixture-of-Experts implementation with load balancing

### 2. **Expert Modules** (`src/experts/`)
Specialized experts for different aspects of chart reasoning:

#### **Visual-Spatial & Structural Experts**
- **LayoutDetectionExpert**: Chart structure and element detection
- **OCRGroundingExpert**: Text extraction and positioning
- **ScaleInterpretationExpert**: Axis scale and coordinate interpretation
- **GeometricPropertyExpert**: Geometric property analysis
- **TrendPatternExpert**: Pattern and trend identification

#### **Semantic & Relational Experts**
- **QueryDeconstructionExpert**: Question understanding and decomposition
- **NumericalReasoningExpert**: Mathematical reasoning and calculations
- **KnowledgeIntegrationExpert**: Information synthesis

#### **Cross-Modal Fusion & Reasoning Experts**
- **VisualTextualAlignmentExpert**: Visual-textual correspondence
- **ChartToGraphExpert**: Chart-to-graph transformation

#### **Cognitive Effort Modulation Experts**
- **ShallowReasoningExpert**: Fast, simple reasoning
- **DeepReasoningOrchestratorExpert**: Complex multi-step reasoning coordination

### 3. **Routing Mechanisms** (`src/routing/`)
- **DynamicRouter**: Content-aware routing with modality detection
- **HierarchicalRouter**: Multi-level routing strategy
- **SkillBasedRouter**: Task-specific expert selection
- **RLRouter**: Reinforcement learning enhanced routing

### 4. **Data Handling** (`src/data/`)
- **ChartMuseumDataset**: ChartMuseum benchmark loader
- **ChartQADataset**: ChartQA dataset loader
- **PlotQADataset**: PlotQA dataset loader
- **MultiDatasetLoader**: Combined dataset management

## 🚀 Getting Started

### Installation
```bash
# Clone the repository
git clone https://github.com/your-org/ChartExpert-MoE.git
cd ChartExpert-MoE

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Quick Usage

#### 1. **Explore ChartMuseum Dataset**
```bash
python examples/load_chartmuseum.py
```

#### 2. **Train ChartExpert-MoE**
```bash
python scripts/train.py --config configs/chart_expert_base.yaml --stage all
```

#### 3. **Run Demo**
```bash
python scripts/demo.py \
    --model_path checkpoints/final_model \
    --image_path path/to/chart.png \
    --question "What is the trend in this data?" \
    --show_expert_analysis
```

## 🎯 Key Features

### **1. Specialized Expert Architecture**
- 10 specialized expert modules for different chart reasoning tasks
- Hierarchical organization: Visual → Semantic → Cross-modal → Cognitive
- Each expert optimized for specific chart understanding challenges

### **2. Advanced Routing Mechanisms**
- Content-aware routing based on input characteristics
- Modality-aware routing (visual vs textual dominance)
- Load balancing to ensure expert diversity
- Support for rule-based, learned, and hybrid routing strategies

### **3. Multi-Stage Training**
- **Stage 1**: Foundation pre-training
- **Stage 2**: Vision-language joint pre-training
- **Stage 3**: Chart-specific instruction tuning
- **Stage 4**: Expert specialization
- **Stage 5**: ChartMuseum fine-tuning

### **4. Comprehensive Evaluation**
- ChartMuseum benchmark integration
- Expert activation analysis
- Reasoning process explanation
- Performance breakdown by chart type and reasoning type

### **5. Flexible Configuration**
- YAML-based configuration system
- Support for different base models
- Configurable expert parameters
- Multi-GPU and distributed training support

## 📊 Supported Datasets

- **ChartMuseum**: Challenging benchmark focusing on complex visual reasoning
- **ChartQA**: Standard chart question answering dataset
- **PlotQA**: Plot-based question answering dataset
- **Custom datasets**: Extensible framework for new datasets

## 🔧 Advanced Features

### **Expert Analysis**
The demo script provides detailed expert activation analysis:
- Which experts were activated for each query
- Reasoning process explanation
- Expected vs actual expert utilization
- Confidence scoring

### **Flexible Base Models**
Support for multiple vision encoders and LLMs:
- **Vision**: CLIP, SigLIP, DINOv2, MoonViT
- **Language**: Llama 3, Qwen2.5-VL, Gemma 3

### **Efficient Training**
- Mixed precision training
- Gradient checkpointing
- DeepSpeed integration
- Wandb logging and monitoring

## 📈 Expected Performance

ChartExpert-MoE is designed to significantly improve performance on:
- Complex visual reasoning tasks (target: 70%+ accuracy on ChartMuseum)
- Multi-step chart reasoning
- Cross-modal understanding
- Trend and pattern analysis
- Numerical computation from visual data

## 🤝 Contributing

Please refer to `docs/CONTRIBUTING.md` for contribution guidelines.

## 📝 License

MIT License - see LICENSE file for details.

## 📚 Citation

```bibtex
@article{chartexpert2024,
  title={ChartExpert-MoE: A Novel MoE-VLM Architecture for Complex Chart Reasoning},
  author={Your Name},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
``` 