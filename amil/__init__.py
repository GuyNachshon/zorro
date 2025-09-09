"""
AMIL (Attention-based Multiple Instance Learning) for Malicious Package Detection.

A lightweight, interpretable alternative to ICN for CI/CD and registry scanning.
Focuses on package-level classification with instance-level explanations.

This is a separate package from ICN, designed for production deployment scenarios
where speed and interpretability are prioritized over the full convergence analysis.
"""

from .model import AMILModel, AMILOutput
from .feature_extractor import AMILFeatureExtractor, UnitFeatures
from .config import AMILConfig, TrainingConfig, EvaluationConfig
from .trainer import AMILTrainer
from .evaluator import AMILEvaluator
from .losses import AMILLossFunction

__version__ = "0.1.0"
__all__ = [
    "AMILModel",
    "AMILOutput", 
    "AMILFeatureExtractor",
    "UnitFeatures",
    "AMILConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "AMILTrainer",
    "AMILEvaluator",
    "AMILLossFunction"
]