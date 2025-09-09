"""
Configuration for CPG-GNN model and training.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class CPGConfig:
    """Configuration for CPG-GNN model architecture."""
    
    # Graph construction
    max_nodes_per_graph: int = 5000
    max_edges_per_graph: int = 20000
    include_ast_edges: bool = True
    include_cfg_edges: bool = True  # Control flow graph
    include_dfg_edges: bool = True  # Data flow graph
    
    # Node features
    node_embedding_dim: int = 512  # CodeBERT embedding dimension
    node_type_vocab_size: int = 100  # Number of AST node types
    use_pretrained_embeddings: bool = True
    pretrained_model_name: str = "microsoft/codebert-base"
    
    # Edge features
    edge_type_vocab_size: int = 10  # AST, CFG, DFG, etc.
    use_edge_features: bool = True
    
    # GNN architecture
    gnn_type: str = "gin"  # Options: "gin", "gat", "graphtransformer"
    num_gnn_layers: int = 3
    gnn_hidden_dim: int = 256
    gnn_dropout: float = 0.2
    
    # Pooling
    pooling_type: str = "attention"  # Options: "mean", "max", "attention"
    attention_heads: int = 4
    
    # Classifier
    classifier_hidden_dim: int = 128
    classifier_dropout: float = 0.3
    num_classes: int = 2  # Binary: malicious/benign
    
    # API detection
    risky_apis: List[str] = field(default_factory=lambda: [
        "subprocess", "eval", "exec", "compile", "__import__",
        "fs", "net", "http", "https", "crypto", "child_process",
        "os.system", "os.popen", "urllib", "requests",
        "base64", "pickle", "marshal"
    ])
    
    # Metadata features
    use_metadata_features: bool = True
    metadata_embedding_dim: int = 32


@dataclass
class TrainingConfig:
    """Configuration for CPG-GNN training."""
    
    # Basic training
    batch_size: int = 16  # Smaller due to graph size
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    gradient_clip_value: float = 1.0
    
    # Curriculum learning stages
    curriculum_stages: Dict[str, Dict] = field(default_factory=lambda: {
        "stage_a": {
            "name": "balanced_training",
            "epochs": 30,
            "malicious_ratio": 0.5,
            "include_trojans": False,
            "augmentation": False
        },
        "stage_b": {
            "name": "trojan_training", 
            "epochs": 25,
            "malicious_ratio": 0.4,
            "include_trojans": True,
            "augmentation": False
        },
        "stage_c": {
            "name": "robustness_training",
            "epochs": 25,
            "malicious_ratio": 0.3,
            "include_trojans": True,
            "augmentation": True  # Obfuscation
        }
    })
    
    # Loss weights
    classification_loss_weight: float = 1.0
    api_prediction_loss_weight: float = 0.2  # Auxiliary task
    entropy_prediction_loss_weight: float = 0.1  # Auxiliary task
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"  # Options: "cosine", "step", "none"
    warmup_epochs: int = 5
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_auc"
    early_stopping_mode: str = "max"
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/cpg"
    save_top_k: int = 3
    
    # Data augmentation
    augmentation_prob: float = 0.3
    obfuscation_types: List[str] = field(default_factory=lambda: [
        "minification",
        "variable_renaming", 
        "string_encoding",
        "control_flow_flattening"
    ])
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class EvaluationConfig:
    """Configuration for CPG-GNN evaluation."""
    
    # Metrics
    primary_metric: str = "roc_auc"
    metrics_to_compute: List[str] = field(default_factory=lambda: [
        "roc_auc", "precision", "recall", "f1", "fpr_at_95_tpr"
    ])
    
    # Thresholds
    target_roc_auc: float = 0.95
    target_fpr: float = 0.02  # Max 2% false positive rate
    target_localization_iou: float = 0.7
    
    # Speed benchmarking
    max_inference_time_seconds: float = 10.0
    benchmark_package_sizes: List[str] = field(default_factory=lambda: [
        "small",   # <10 files
        "medium",  # 10-50 files
        "large"    # 50+ files
    ])
    
    # Robustness testing
    obfuscation_drop_threshold: float = 0.05  # Max 5% performance drop
    test_obfuscation_types: List[str] = field(default_factory=lambda: [
        "minification",
        "base64_encoding",
        "variable_renaming",
        "dead_code_injection"
    ])
    
    # Localization analysis
    compute_attention_maps: bool = True
    top_k_subgraphs: int = 5
    min_subgraph_size: int = 3  # Minimum nodes in suspicious subgraph
    
    # Interpretability
    generate_explanations: bool = True
    highlight_api_paths: bool = True  # Show env→encode→net paths
    
    # Batch evaluation
    eval_batch_size: int = 8
    use_multiprocessing: bool = True


def create_default_config() -> tuple[CPGConfig, TrainingConfig, EvaluationConfig]:
    """Create default configuration for CPG-GNN."""
    return CPGConfig(), TrainingConfig(), EvaluationConfig()


def load_config_from_json(config_path: str) -> tuple[CPGConfig, TrainingConfig, EvaluationConfig]:
    """Load configuration from JSON file."""
    import json
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    cpg_config = CPGConfig(**config_dict.get('cpg', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    eval_config = EvaluationConfig(**config_dict.get('evaluation', {}))
    
    return cpg_config, training_config, eval_config


def save_config_to_json(
    cpg_config: CPGConfig,
    training_config: TrainingConfig,
    eval_config: EvaluationConfig,
    save_path: str
):
    """Save configuration to JSON file."""
    import json
    from dataclasses import asdict
    
    config_dict = {
        'cpg': asdict(cpg_config),
        'training': asdict(training_config),
        'evaluation': asdict(eval_config)
    }
    
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)