"""
Zorro Evaluation Package

Unified evaluation system for malicious package detection models.
Supports multiple model types, prompt strategies, and configuration management.
"""

from .config import EvaluationConfig, load_config
from .runner import EvaluationRunner
from .results import ResultsAnalyzer

__version__ = "1.0.0"
__all__ = ["EvaluationConfig", "load_config", "EvaluationRunner", "ResultsAnalyzer"]