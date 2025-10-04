"""
Configuration for NeoBERT classifier.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class NeoBERTConfig:
    """Configuration for NeoBERT model architecture."""
    
    # Model selection
    model_name: str = "microsoft/codebert-base"  # Fallback to CodeBERT if NeoBERT unavailable
    use_pretrained: bool = True
    freeze_encoder: bool = False  # Whether to freeze NeoBERT weights initially
    
    # Tokenization
    max_tokens_per_unit: int = 512
    tokenizer_max_length: int = 512
    unit_chunking_strategy: str = "sliding_window"  # Options: "truncate", "sliding_window", "function_aware"
    chunk_overlap: int = 50  # Tokens overlap between chunks
    
    # Unit processing
    unit_type: str = "file"  # Options: "file", "function", "mixed"
    max_units_per_package: int = 100
    min_unit_tokens: int = 10  # Skip units shorter than this
    
    # Embeddings
    embedding_dim: int = 768  # NeoBERT/CodeBERT embedding dimension
    projection_dim: int = 512  # Projected embedding dimension
    use_augmented_features: bool = True
    
    # Augmented features
    risky_apis: List[str] = field(default_factory=lambda: [
        "subprocess", "eval", "exec", "compile", "__import__", "open",
        "fs", "net", "http", "https", "crypto", "child_process",
        "os.system", "os.popen", "urllib", "requests", "fetch",
        "base64", "pickle", "marshal", "xmlrpc"
    ])
    
    # Phase detection patterns
    install_phase_patterns: List[str] = field(default_factory=lambda: [
        "postinstall", "preinstall", "install", "setup.py", "setup.cfg"
    ])
    
    test_phase_patterns: List[str] = field(default_factory=lambda: [
        "test", "spec", "__test__", "tests/"
    ])
    
    # Feature dimensions
    api_feature_dim: int = 32
    entropy_feature_bins: int = 10
    phase_feature_dim: int = 8
    metadata_feature_dim: int = 16
    
    # Pooling strategy
    pooling_strategy: str = "attention"  # Options: "mean", "attention", "mil"
    attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # MIL-specific settings (if using MIL pooling)
    mil_hidden_dim: int = 256
    mil_attention_dim: int = 128
    
    # Classifier
    classifier_hidden_dim: int = 128
    classifier_dropout: float = 0.3
    num_classes: int = 2  # Binary classification
    
    # Auxiliary tasks
    use_api_prediction: bool = True
    use_phase_prediction: bool = True
    api_prediction_threshold: float = 0.5
    
    # Device and performance
    device: str = "auto"
    batch_size: int = 8  # Reduced for package-level training (use gradient accumulation)
    gradient_checkpointing: bool = True  # Save memory
    mixed_precision: bool = True  # Use fp16 training


@dataclass  
class TrainingConfig:
    """Configuration for NeoBERT training."""
    
    # Curriculum learning stages
    curriculum_stages: Dict[str, Dict] = field(default_factory=lambda: {
        "stage_a": {
            "name": "small_balanced",
            "epochs": 5,
            "benign_samples": 5000,
            "malicious_samples": 1000,
            "augmentation": False,
            "hard_negatives": False
        },
        "stage_b": {
            "name": "scaled_dataset", 
            "epochs": 4,
            "benign_samples": 25000,
            "malicious_samples": 4000,
            "augmentation": False,
            "hard_negatives": True
        },
        "stage_c": {
            "name": "robustness_training",
            "epochs": 3,
            "benign_samples": 30000,
            "malicious_samples": 5000,
            "augmentation": True,
            "hard_negatives": True
        }
    })
    
    # Basic training
    batch_size: int = 8  # With 2 GPUs (DataParallel), can handle larger batches
    max_packages_per_batch: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    
    # Learning rate scheduling
    warmup_steps: int = 1000
    scheduler_type: str = "cosine"  # Options: "cosine", "linear", "constant"
    
    # Loss function weights
    classification_loss_weight: float = 1.0
    api_prediction_loss_weight: float = 0.2
    phase_prediction_loss_weight: float = 0.1
    
    # Cost-sensitive learning (for imbalanced data)
    use_class_weights: bool = True
    pos_weight: float = 5.0  # Weight for positive (malicious) class
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    use_focal_loss: bool = False
    
    # Data augmentation
    augmentation_prob: float = 0.3
    augmentation_types: List[str] = field(default_factory=lambda: [
        "minification",
        "base64_encoding", 
        "variable_renaming",
        "string_obfuscation",
        "comment_removal"
    ])
    
    # Hard negatives (benign packages with risky APIs)
    hard_negative_ratio: float = 0.2  # Portion of benign samples that are hard negatives
    hard_negative_api_threshold: int = 3  # Min risky APIs for hard negative
    
    # Validation and checkpointing
    validation_split: float = 0.15
    evaluation_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_auc"
    early_stopping_mode: str = "max"
    
    # Optimizer
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Hardware and performance
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    enable_gradient_checkpointing: bool = True
    
    # Logging and monitoring
    log_every_n_steps: int = 100
    use_wandb: bool = True
    wandb_project: str = "neobert-malware-detection"
    
    # Checkpoints
    checkpoint_dir: str = "checkpoints/neobert"
    resume_from_checkpoint: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Configuration for NeoBERT evaluation."""
    
    # Primary metrics
    target_roc_auc: float = 0.95
    target_fpr_at_95_tpr: float = 0.02  # 2% FPR at 95% TPR
    target_precision: float = 0.90
    target_recall: float = 0.95
    
    # Speed benchmarks
    target_inference_time_small: float = 2.0  # seconds for <10 files
    target_inference_time_large: float = 5.0  # seconds for large packages
    package_size_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "small": 10,    # <10 files
        "medium": 50,   # 10-50 files  
        "large": 100    # 50+ files
    })
    
    # Robustness testing
    robustness_drop_threshold: float = 0.05  # Max 5% performance drop
    obfuscation_types: List[str] = field(default_factory=lambda: [
        "minification",
        "base64_strings",
        "variable_renaming", 
        "dead_code_injection",
        "control_flow_obfuscation"
    ])
    
    # Generalization testing
    zero_day_families: List[str] = field(default_factory=lambda: [
        "credential_stealers",
        "crypto_miners", 
        "backdoors",
        "data_exfiltrators"
    ])
    family_holdout_ratio: float = 0.2  # Hold out 20% of families for zero-day testing
    
    # Localization evaluation (for attention/MIL models)
    evaluate_localization: bool = True
    localization_iou_threshold: float = 0.7
    synthetic_trojan_samples: int = 100
    
    # Interpretability
    generate_explanations: bool = True
    max_suspicious_units: int = 5
    attention_threshold: float = 0.1  # Min attention weight to consider significant
    
    # Batch evaluation
    eval_batch_size: int = 16
    use_caching: bool = True  # Cache embeddings for repeated evaluation
    
    # Comparison with other models
    compare_with_baselines: bool = True
    baseline_models: List[str] = field(default_factory=lambda: [
        "ICN", "AMIL", "CPG-GNN"
    ])
    
    # Threshold calibration
    calibrate_threshold: bool = True
    calibration_methods: List[str] = field(default_factory=lambda: [
        "platt_scaling",
        "isotonic_regression"
    ])
    
    # Output formats
    save_predictions: bool = True
    save_attention_maps: bool = True  # For attention/MIL models
    generate_report: bool = True
    
    # Statistical testing
    confidence_interval: float = 0.95
    significance_threshold: float = 0.05
    bootstrap_samples: int = 1000


def create_default_config() -> tuple[NeoBERTConfig, TrainingConfig, EvaluationConfig]:
    """Create default configuration for NeoBERT classifier."""
    return NeoBERTConfig(), TrainingConfig(), EvaluationConfig()


def load_config_from_json(config_path: str) -> tuple[NeoBERTConfig, TrainingConfig, EvaluationConfig]:
    """Load configuration from JSON file."""
    import json
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    neobert_config = NeoBERTConfig(**config_dict.get('neobert', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    eval_config = EvaluationConfig(**config_dict.get('evaluation', {}))
    
    return neobert_config, training_config, eval_config


def save_config_to_json(
    neobert_config: NeoBERTConfig,
    training_config: TrainingConfig, 
    eval_config: EvaluationConfig,
    save_path: str
):
    """Save configuration to JSON file."""
    import json
    from dataclasses import asdict
    
    config_dict = {
        'neobert': asdict(neobert_config),
        'training': asdict(training_config),
        'evaluation': asdict(eval_config)
    }
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def get_model_name_from_config(config: NeoBERTConfig) -> str:
    """Get appropriate model name, with fallback handling."""
    
    # Try to use NeoBERT if available, otherwise fall back to CodeBERT
    preferred_models = [
        "microsoft/neobert-base",  # Hypothetical NeoBERT model
        "microsoft/codebert-base",
        "microsoft/graphcodebert-base"
    ]
    
    if config.model_name in preferred_models:
        return config.model_name
    else:
        # Default fallback
        return "microsoft/codebert-base"


def validate_config(config: NeoBERTConfig) -> List[str]:
    """Validate configuration and return list of issues."""
    issues = []
    
    if config.max_units_per_package < 1:
        issues.append("max_units_per_package must be >= 1")
    
    if config.max_tokens_per_unit < 10:
        issues.append("max_tokens_per_unit too small (< 10)")
        
    if config.max_tokens_per_unit > 512:
        issues.append("max_tokens_per_unit exceeds transformer limit (512)")
        
    if config.projection_dim < 64:
        issues.append("projection_dim too small (< 64)")
        
    if config.pooling_strategy not in ["mean", "attention", "mil"]:
        issues.append(f"Invalid pooling_strategy: {config.pooling_strategy}")
        
    if config.attention_heads < 1:
        issues.append("attention_heads must be >= 1")
        
    if config.classifier_dropout < 0 or config.classifier_dropout > 1:
        issues.append("classifier_dropout must be in [0, 1]")
    
    return issues