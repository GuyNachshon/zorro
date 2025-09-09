"""
NeoBERT: Modern transformer-based classifier for malicious package detection.

This package implements a scalable NeoBERT-based system for detecting malicious
packages using unit-level processing with multiple pooling strategies.
"""

from .model import NeoBERTClassifier, create_neobert_model
from .encoder import NeoBERTEncoder
from .pooling import MeanPooling, AttentionPooling, MILPooling
from .unit_processor import UnitProcessor, PackageUnit
from .config import NeoBERTConfig, TrainingConfig, EvaluationConfig, create_default_config

__all__ = [
    'NeoBERTClassifier',
    'create_neobert_model',
    'NeoBERTEncoder',
    'MeanPooling',
    'AttentionPooling', 
    'MILPooling',
    'UnitProcessor',
    'PackageUnit',
    'NeoBERTConfig',
    'TrainingConfig',
    'EvaluationConfig',
    'create_default_config'
]

__version__ = '0.1.0'