# ChartExpert-MoE: A Novel MoE-VLM Architecture for Complex Chart Reasoning

## Overview

ChartExpert-MoE is a specialized Mixture-of-Experts Vision-Language Model designed to address the "visual reasoning gap" in chart understanding. Unlike existing VLMs that often rely heavily on textual cues, our architecture employs specialized expert modules to handle different aspects of chart comprehension, from visual-spatial analysis to complex multi-step reasoning.

## Key Features

- **Specialized Expert Modules**: Fine-grained experts for visual-spatial analysis, semantic reasoning, and cross-modal fusion
- **Dynamic Routing Mechanisms**: Intelligent routing strategies including hierarchical, skill-based, and RL-enhanced routing
- **Advanced Vision-Text Fusion**: Dynamic gated fusion with structural chart information integration
- **Multi-stage Training**: Progressive training from general VLM to chart-specific specialization
- **ChartMuseum Optimization**: Specifically designed to address challenges identified in the ChartMuseum benchmark

## Architecture Components

### Expert Modules
1. **Visual-Spatial & Structural Experts**
   - Layout & Element Detection Expert
   - OCR & Text Grounding Expert
   - Scale & Coordinate Interpretation Expert
   - Geometric Property Expert
   - Trend & Pattern Perception Expert

2. **Semantic & Relational Experts**
   - Query Deconstruction & Intent Expert
   - Numerical & Logical Reasoning Expert
   - Knowledge Integration Expert

3. **Cross-Modal Fusion & Reasoning Experts**
   - Visual-Textual Alignment Expert
   - Chart-to-Graph Transformation Expert

4. **Cognitive Effort Modulation Experts**
   - Shallow Reasoning Expert
   - Deep Reasoning Orchestrator Expert

### Base Models Support
- **Vision Encoders**: MoonViT, DINOv2, SAM
- **LLMs**: Llama 3 series, Qwen2.5-VL, Gemma 3

## Installation

```bash
git clone https://github.com/your-org/ChartExpert-MoE.git
cd ChartExpert-MoE
pip install -r requirements.txt
```

## Quick Start

```python
from src.models.chart_expert_moe import ChartExpertMoE
from src.data.datasets import ChartMuseumDataset

# Initialize model
model = ChartExpertMoE(config_path="configs/chart_expert_base.yaml")

# Load ChartMuseum dataset
dataset = ChartMuseumDataset()

# Run inference
result = model.predict(image_path="path/to/chart.png", query="What is the trend in the data?")
```

## Training

```bash
# Stage 1: Foundation pre-training
python scripts/train.py --stage foundation --config configs/stage1_foundation.yaml

# Stage 2: Vision-language joint pre-training
python scripts/train.py --stage joint_pretrain --config configs/stage2_joint.yaml

# Stage 3: Chart-specific instruction tuning
python scripts/train.py --stage chart_tuning --config configs/stage3_chart.yaml

# Stage 4: Expert specialization
python scripts/train.py --stage expert_specialization --config configs/stage4_experts.yaml

# Stage 5: ChartMuseum fine-tuning
python scripts/train.py --stage chartmuseum_finetune --config configs/stage5_chartmuseum.yaml
```

## Evaluation

```bash
# Evaluate on ChartMuseum
python scripts/evaluate.py --dataset chartmuseum --model_path checkpoints/best_model.pt

# Run ablation studies
python scripts/ablation.py --config configs/ablation_experts.yaml
```

## Citation

```bibtex
@article{chartexpert2024,
  title={ChartExpert-MoE: A Novel MoE-VLM Architecture for Complex Chart Reasoning},
  author={Your Name},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## License

MIT License

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project. 