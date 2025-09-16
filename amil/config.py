"""
Configuration classes for AMIL (Attention-based Multiple Instance Learning).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch


@dataclass
class AMILConfig:
    """Core AMIL model configuration."""
    
    # Model architecture
    code_embedding_dim: int = 768  # GraphCodeBERT/CodeBERT dimension
    api_feature_dim: int = 20  # Number of API categories tracked
    entropy_feature_dim: int = 5  # Entropy and obfuscation features
    phase_feature_dim: int = 3  # install, runtime, test phases
    metadata_feature_dim: int = 10  # File size, imports, functions, etc.
    
    # Feature fusion
    unit_embedding_dim: int = 512  # Final unit embedding dimension
    dropout_rate: float = 0.2
    
    # Attention-MIL pooling
    attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # Package classifier
    classifier_hidden_dim: int = 128
    classifier_dropout: float = 0.3
    
    # Batch processing limits
    max_units_per_package: int = 100  # Cap for performance
    max_packages_per_batch: int = 32
    
    # API categories for feature extraction
    api_categories: List[str] = field(default_factory=lambda: [
        "net.outbound", "net.inbound", "fs.read", "fs.write", "fs.delete",
        "crypto.hash", "crypto.encrypt", "subprocess.spawn", "subprocess.shell",
        "eval.exec", "eval.compile", "env.read", "env.write", "install.hook",
        "install.script", "obfuscation.base64", "obfuscation.eval", 
        "registry.access", "browser.access", "system.info"
    ])
    
    # Entropy thresholds for suspicious content detection
    entropy_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 3.0,      # Normal text
        "medium": 4.5,   # Potentially encoded
        "high": 6.0,     # Likely obfuscated
        "very_high": 7.0 # Almost certainly encoded/compressed
    })
    
    def total_feature_dim(self) -> int:
        """Calculate total input feature dimension."""
        return (self.code_embedding_dim + 
                self.api_feature_dim + 
                self.entropy_feature_dim + 
                self.phase_feature_dim + 
                self.metadata_feature_dim)


@dataclass 
class TrainingConfig:
    """Training configuration for AMIL curriculum learning."""
    
    # Optimizer settings
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"  # adamw, adam, sgd
    
    # Curriculum stages (as per AMIL.md specification)
    curriculum_stages: Dict[str, Dict] = field(default_factory=lambda: {
        "stage_a": {
            "name": "balanced_training",
            "malicious_ratio": 0.2,  # 1:5 malicious:benign ratio
            "epochs": 20,
            "augmentation": False,
            "description": "Basic balanced training"
        },
        "stage_b": {
            "name": "augmented_training", 
            "malicious_ratio": 0.2,
            "epochs": 15,
            "augmentation": True,
            "obfuscation_types": ["minify", "base64", "string_split"],
            "description": "Add obfuscated variants"
        },
        "stage_c": {
            "name": "realistic_training",
            "malicious_ratio": 0.1,  # 1:10 ratio for real-world calibration
            "epochs": 25,
            "augmentation": True,
            "description": "Real-world ratio training"
        }
    })
    
    # Loss function weights
    bce_weight: float = 1.0  # Binary cross-entropy
    sparsity_weight: float = 0.01  # Attention sparsity regularization
    counterfactual_weight: float = 0.05  # Counterfactual consistency
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    monitor_metric: str = "val_roc_auc"  # val_roc_auc, val_pr_auc, val_loss
    
    # Batch and sampling
    batch_size: int = 16  # Packages per batch
    validation_split: float = 0.2
    test_split: float = 0.2
    
    # Data augmentation parameters
    augmentation_config: Dict[str, Dict] = field(default_factory=lambda: {
        "minify": {
            "probability": 0.3,
            "remove_whitespace": True,
            "remove_comments": True,
            "shorten_names": False  # Keep for interpretability
        },
        "base64": {
            "probability": 0.2,
            "encode_strings": True,
            "encode_full_functions": False  # Too aggressive
        },
        "string_split": {
            "probability": 0.25,
            "max_splits": 3,
            "min_length": 10
        }
    })
    
    # Checkpointing
    save_every_n_epochs: int = 5
    save_best_only: bool = True
    checkpoint_dir: str = "checkpoints/amil"


@dataclass
class EvaluationConfig:
    """Evaluation and metrics configuration."""
    
    # Classification metrics
    target_roc_auc: float = 0.95  # Success criterion from AMIL.md
    target_fpr_at_95tpr: float = 0.02  # <2% false positive rate
    
    # Localization metrics (for synthetic trojans)
    target_localization_iou: float = 0.7  # Attention IoU with ground truth
    target_counterfactual_drop: float = 0.5  # Score drop when masking top unit
    top_k_attention: int = 3  # Analyze top-K attended units
    
    # Speed benchmarking
    target_inference_latency: float = 2.0  # ≤2s per package
    speed_test_samples: int = 100
    
    # Attention analysis
    attention_percentiles: List[float] = field(default_factory=lambda: [50, 75, 90, 95, 99])
    visualize_attention: bool = True
    save_attention_maps: bool = True
    
    # Robustness testing
    obfuscation_test_types: List[str] = field(default_factory=lambda: [
        "minified", "base64_encoded", "string_split", "variable_renamed"
    ])
    
    # Cross-ecosystem testing
    test_cross_ecosystem: bool = True  # Train on npm, test on PyPI and vice versa
    
    # Statistical testing
    confidence_interval: float = 0.95
    bootstrap_samples: int = 1000
    significance_level: float = 0.05


@dataclass
class ProductionConfig:
    """Configuration for production deployment."""
    
    # Performance settings
    max_concurrent_packages: int = 4
    timeout_per_package: float = 5.0  # seconds
    memory_limit_mb: int = 2048
    
    # Caching
    enable_embedding_cache: bool = True
    cache_size_mb: int = 512
    cache_ttl_hours: int = 24
    
    # Output formatting
    include_attention_weights: bool = True
    max_attention_units: int = 10
    include_api_breakdown: bool = True
    explanation_detail_level: str = "medium"  # minimal, medium, detailed
    
    # Alerting thresholds
    high_risk_threshold: float = 0.9
    medium_risk_threshold: float = 0.7  
    low_risk_threshold: float = 0.3
    
    # Integration settings
    return_format: str = "json"  # json, protobuf
    enable_metrics_logging: bool = True
    enable_prediction_logging: bool = False  # Privacy consideration


def create_default_config() -> Tuple[AMILConfig, TrainingConfig, EvaluationConfig]:
    """Create default configuration tuple for AMIL."""
    return AMILConfig(), TrainingConfig(), EvaluationConfig()


def validate_config(amil_config: AMILConfig, training_config: TrainingConfig) -> bool:
    """Validate configuration compatibility and constraints."""
    
    # Check dimension compatibility
    total_dim = amil_config.total_feature_dim()
    if total_dim != amil_config.code_embedding_dim + amil_config.api_feature_dim + \
                    amil_config.entropy_feature_dim + amil_config.phase_feature_dim + \
                    amil_config.metadata_feature_dim:
        return False
    
    # Check curriculum stage consistency
    total_epochs = sum(stage["epochs"] for stage in training_config.curriculum_stages.values())
    if total_epochs > 100:  # Reasonable upper bound
        print(f"Warning: Total training epochs ({total_epochs}) is quite high")
    
    # Check loss weights sum to reasonable range
    total_loss_weight = (training_config.bce_weight + 
                        training_config.sparsity_weight + 
                        training_config.counterfactual_weight)
    if total_loss_weight < 0.5 or total_loss_weight > 2.0:
        print(f"Warning: Total loss weights ({total_loss_weight:.3f}) outside recommended range [0.5, 2.0]")
    
    return True


def save_config_to_json(amil_config: AMILConfig, training_config: TrainingConfig,
                       eval_config: EvaluationConfig, filepath: str):
    """Save configuration to JSON file."""
    import json
    from dataclasses import asdict

    config_dict = {
        "amil_config": asdict(amil_config),
        "training_config": asdict(training_config),
        "evaluation_config": asdict(eval_config)
    }

    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)


def load_config_from_json(filepath: str) -> Tuple[AMILConfig, TrainingConfig, EvaluationConfig]:
    """Load configuration from JSON file."""
    import json

    with open(filepath, 'r') as f:
        config_dict = json.load(f)

    amil_config = AMILConfig(**config_dict["amil_config"])
    training_config = TrainingConfig(**config_dict["training_config"])
    eval_config = EvaluationConfig(**config_dict["evaluation_config"])

    return amil_config, training_config, eval_config