"""
CPG-GNN: Code Property Graph with Graph Neural Networks for Malicious Package Detection.

This package implements a graph-based detector that captures structural and semantic flows
in code to identify malicious packages through graph neural networks.
"""

from .model import CPGModel, create_cpg_model
from .graph_builder import CPGBuilder, CodePropertyGraph
from .feature_extractor import CPGFeatureExtractor
from .config import CPGConfig, TrainingConfig, EvaluationConfig, create_default_config

__all__ = [
    'CPGModel',
    'create_cpg_model',
    'CPGBuilder',
    'CodePropertyGraph',
    'CPGFeatureExtractor',
    'CPGConfig',
    'TrainingConfig',
    'EvaluationConfig',
    'create_default_config'
]

__version__ = '0.1.0'