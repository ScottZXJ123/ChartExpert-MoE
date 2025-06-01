"""
ChartExpert-MoE: A Novel MoE-VLM Architecture for Complex Chart Reasoning
"""

__version__ = "0.1.0"
__author__ = "ChartExpert-MoE Team"

from .models import ChartExpertMoE
from .data import ChartMuseumDataset

__all__ = ["ChartExpertMoE", "ChartMuseumDataset"] 