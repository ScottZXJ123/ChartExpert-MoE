"""
Utility functions for ChartExpert-MoE
"""

import logging
import os
import yaml
from typing import Dict, Any
from .model_utils import count_parameters, get_model_size
from .data_utils import collate_fn, prepare_batch


def setup_logging(log_level: str = "INFO", log_dir: str = "./logs"):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Setup logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'training.log'))
        ]
    )


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, indent=2)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


__all__ = [
    "setup_logging",
    "save_config", 
    "load_config",
    "count_parameters",
    "get_model_size",
    "collate_fn",
    "prepare_batch"
] 