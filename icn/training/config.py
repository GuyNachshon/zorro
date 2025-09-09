"""
Training configuration for ICN curriculum learning.
Defines all hyperparameters and stage-specific settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import torch
from pathlib import Path


class OptimizerType(Enum):
    """Supported optimizer types."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class SchedulerType(Enum):
    """Supported learning rate schedulers."""
    LINEAR = "linear"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # Different LRs for different components
    encoder_lr: Optional[float] = None  # Transformer encoder
    head_lr: Optional[float] = None     # Classification heads
    global_lr: Optional[float] = None   # Global integrator


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: SchedulerType = SchedulerType.LINEAR
    warmup_steps: int = 1000
    total_steps: Optional[int] = None
    
    # Scheduler-specific parameters
    gamma: float = 0.95  # For exponential
    patience: int = 5    # For reduce_on_plateau
    factor: float = 0.5  # For reduce_on_plateau


@dataclass
class TrainingConfig:
    """Main training configuration."""
    
    # Model architecture
    model_name: str = "microsoft/codebert-base"
    use_pretrained: bool = True
    embedding_dim: int = 768
    hidden_dim: int = 512
    n_fixed_intents: int = 15
    n_latent_intents: int = 10
    max_seq_length: int = 512
    max_units_per_package: int = 50
    max_convergence_iterations: int = 6
    convergence_threshold: float = 0.01
    
    # Training parameters
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    effective_batch_size: int = field(init=False)
    max_epochs: int = 10
    max_steps: Optional[int] = None
    
    # Optimization
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    grad_clip_norm: float = 1.0
    
    # Mixed precision training
    use_mixed_precision: bool = True
    fp16: bool = True
    
    # Device and parallel training
    device: str = "auto"  # auto, cpu, cuda, or specific GPU
    n_gpu: int = 1
    local_rank: int = -1  # For distributed training
    
    # Checkpointing and saving
    output_dir: str = "checkpoints/icn"
    save_steps: int = 500
    save_total_limit: int = 5
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_roc_auc"
    greater_is_better: bool = True
    
    # Evaluation
    evaluation_strategy: str = "steps"  # steps, epoch, no
    eval_steps: int = 500
    eval_accumulation_steps: Optional[int] = None
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.001
    
    # Logging
    logging_steps: int = 50
    log_level: str = "info"
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    
    # Data loading
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False
    
    # Curriculum learning
    curriculum_enabled: bool = True
    stage_transition_metric: str = "convergence_stability"
    stage_transition_threshold: float = 0.95
    
    # Loss weights (will be tuned per stage)
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'intent_supervision': 1.0,
        'convergence': 1.0,
        'divergence_margin': 2.0,
        'plausibility': 2.0,
        'classification': 1.0,
        'latent_contrastive': 0.5
    })
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    
    # Data ratios
    benign_malicious_ratio: float = 5.0  # 5:1 benign:malicious
    malicious_batch_ratio: float = 0.2   # 20% malicious per batch
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        # Compute effective batch size
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps * self.n_gpu
        
        # Set scheduler total steps if not provided
        if self.scheduler.total_steps is None and self.max_steps:
            self.scheduler.total_steps = self.max_steps


@dataclass
class CurriculumStageConfig:
    """Configuration for a specific curriculum learning stage."""
    
    name: str
    description: str
    
    # Stage-specific training parameters
    max_epochs: int = 5
    max_steps: Optional[int] = None
    learning_rate: float = 2e-5
    batch_size: Optional[int] = None  # Use default if None
    
    # Loss configuration
    active_losses: List[str] = field(default_factory=list)
    loss_weights: Dict[str, float] = field(default_factory=dict)
    
    # Data filtering
    allowed_sample_types: List[str] = field(default_factory=lambda: ["benign"])
    balanced_sampling: bool = False
    
    # Evaluation
    eval_metric: str = "loss"
    success_threshold: float = 0.0
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.001
    
    # Special configurations
    freeze_encoder: bool = False
    augmentation: bool = False
    
    # Stage transition criteria
    transition_criteria: Dict[str, float] = field(default_factory=dict)


class CurriculumConfig:
    """Defines the complete curriculum learning configuration."""
    
    def __init__(self):
        self.stages = self._create_curriculum_stages()
    
    def _create_curriculum_stages(self) -> Dict[str, CurriculumStageConfig]:
        """Create configurations for all curriculum stages."""
        
        stages = {}
        
        # Stage A: Pretraining on benign samples
        stages["stage_a_pretraining"] = CurriculumStageConfig(
            name="Stage A: Pretraining",
            description="Pretrain Local Intent Estimator on benign packages for intent prediction",
            max_epochs=3,
            learning_rate=2e-5,
            active_losses=["intent_supervision", "latent_contrastive"],
            loss_weights={
                "intent_supervision": 2.0,
                "latent_contrastive": 1.0
            },
            allowed_sample_types=["benign"],
            balanced_sampling=False,
            eval_metric="intent_accuracy",
            success_threshold=0.75,
            patience=3,
            transition_criteria={
                "intent_accuracy": 0.75,
                "loss_stability": 0.01
            }
        )
        
        # Stage B: Convergence training on benign samples
        stages["stage_b_convergence"] = CurriculumStageConfig(
            name="Stage B: Convergence",
            description="Train Global Intent Integrator for stable convergence on benign packages",
            max_epochs=4,
            learning_rate=1e-5,  # Lower LR for stability
            active_losses=["convergence", "intent_supervision"],
            loss_weights={
                "convergence": 3.0,
                "intent_supervision": 1.0
            },
            allowed_sample_types=["benign"],
            balanced_sampling=False,
            eval_metric="convergence_stability",
            success_threshold=0.9,
            patience=5,
            transition_criteria={
                "convergence_stability": 0.9,
                "avg_iterations": 3.5  # Should converge quickly
            }
        )
        
        # Stage C: Add malicious samples
        stages["stage_c_malicious"] = CurriculumStageConfig(
            name="Stage C: Malicious Detection",
            description="Add malicious samples and train dual detection channels",
            max_epochs=6,
            learning_rate=5e-6,  # Even lower for stability
            active_losses=[
                "intent_supervision", "convergence", "divergence_margin", 
                "plausibility", "classification"
            ],
            loss_weights={
                "intent_supervision": 1.0,
                "convergence": 1.0,
                "divergence_margin": 3.0,
                "plausibility": 3.0,
                "classification": 2.0
            },
            allowed_sample_types=["benign", "compromised_lib", "malicious_intent"],
            balanced_sampling=True,
            eval_metric="roc_auc",
            success_threshold=0.85,
            patience=8,
            transition_criteria={
                "roc_auc": 0.85,
                "precision": 0.8,
                "recall": 0.8
            }
        )
        
        # Stage D: Robustness training
        stages["stage_d_robustness"] = CurriculumStageConfig(
            name="Stage D: Robustness",
            description="Robustness training with obfuscated and augmented samples",
            max_epochs=4,
            learning_rate=1e-6,  # Very low for fine-tuning
            active_losses=[
                "intent_supervision", "convergence", "divergence_margin",
                "plausibility", "classification", "latent_contrastive"
            ],
            loss_weights={
                "intent_supervision": 0.5,
                "convergence": 1.0,
                "divergence_margin": 2.0,
                "plausibility": 2.0,
                "classification": 3.0,  # Focus on final performance
                "latent_contrastive": 0.5
            },
            allowed_sample_types=["benign", "compromised_lib", "malicious_intent"],
            balanced_sampling=True,
            augmentation=True,  # Enable data augmentation
            eval_metric="roc_auc",
            success_threshold=0.9,
            patience=6,
            transition_criteria={
                "roc_auc": 0.9,
                "precision": 0.85,
                "recall": 0.85,
                "robustness_score": 0.8
            }
        )
        
        return stages
    
    def get_stage_config(self, stage_name: str) -> CurriculumStageConfig:
        """Get configuration for a specific stage."""
        return self.stages[stage_name]
    
    def get_stage_names(self) -> List[str]:
        """Get ordered list of stage names."""
        return [
            "stage_a_pretraining",
            "stage_b_convergence", 
            "stage_c_malicious",
            "stage_d_robustness"
        ]


def create_training_config(
    experiment_name: str,
    output_dir: Optional[str] = None,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_epochs: int = 10,
    use_gpu: bool = None,
    **kwargs
) -> TrainingConfig:
    """Create a training configuration with sensible defaults."""
    
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    
    if output_dir is None:
        output_dir = f"checkpoints/{experiment_name}"
    
    # Create base config
    config = TrainingConfig(
        batch_size=batch_size,
        max_epochs=max_epochs,
        output_dir=output_dir,
        device="cuda" if use_gpu else "cpu",
        n_gpu=torch.cuda.device_count() if use_gpu else 1,
        use_mixed_precision=use_gpu,  # Only use mixed precision with GPU
        fp16=use_gpu
    )
    
    # Update optimizer learning rate
    config.optimizer.learning_rate = learning_rate
    
    # Apply any additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def validate_config(config: TrainingConfig) -> List[str]:
    """Validate training configuration and return list of issues."""
    issues = []
    
    # Check batch size
    if config.batch_size <= 0:
        issues.append("batch_size must be positive")
    
    # Check learning rate
    if config.optimizer.learning_rate <= 0:
        issues.append("learning_rate must be positive")
    
    # Check device availability
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        issues.append("CUDA not available but cuda device specified")
    
    # Check output directory
    output_path = Path(config.output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except:
        issues.append(f"Cannot create output directory: {config.output_dir}")
    
    # Check loss weights
    if any(w <= 0 for w in config.loss_weights.values()):
        issues.append("All loss weights must be positive")
    
    # Check curriculum configuration
    if config.curriculum_enabled:
        curriculum = CurriculumConfig()
        stage_names = curriculum.get_stage_names()
        if not stage_names:
            issues.append("No curriculum stages defined")
    
    return issues


if __name__ == "__main__":
    # Test configuration creation and validation
    print("🧪 Testing ICN Training Configuration...")
    
    # Create default config
    config = create_training_config(
        experiment_name="icn_test",
        batch_size=16,
        learning_rate=1e-5,
        max_epochs=5
    )
    
    print(f"✅ Training config created:")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.optimizer.learning_rate}")
    print(f"   Device: {config.device}")
    print(f"   Output dir: {config.output_dir}")
    print(f"   Mixed precision: {config.use_mixed_precision}")
    
    # Test validation
    issues = validate_config(config)
    if issues:
        print(f"❌ Configuration issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ Configuration validation passed")
    
    # Test curriculum configuration
    curriculum = CurriculumConfig()
    print(f"\n📚 Curriculum stages:")
    for stage_name in curriculum.get_stage_names():
        stage_config = curriculum.get_stage_config(stage_name)
        print(f"   {stage_config.name}: {stage_config.description}")
        print(f"     Max epochs: {stage_config.max_epochs}")
        print(f"     Active losses: {stage_config.active_losses}")
        print(f"     Sample types: {stage_config.allowed_sample_types}")
    
    print("\n🚀 Training configuration ready!")