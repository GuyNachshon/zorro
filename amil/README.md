# 🎯 AMIL: Attention-based Multiple Instance Learning for Malicious Package Detection

AMIL is a lightweight, interpretable classifier designed for fast malicious package detection in CI/CD pipelines and registry scanning environments. It uses attention-based multiple instance learning to analyze packages as "bags" of code units, making classifications in a single forward pass while providing interpretable explanations.

## 🏗️ Architecture Overview

### Core Components

1. **Feature Extractor** (`feature_extractor.py`)
   - Combines code embeddings (GraphCodeBERT) with handcrafted features
   - Extracts API patterns, entropy measures, phase analysis, and metadata
   - Produces rich unit-level representations

2. **AMIL Model** (`model.py`)
   - Multi-head attention pooling aggregates unit embeddings
   - Binary classifier with confidence scoring
   - Attention weights provide interpretability

3. **Training Pipeline** (`trainer.py`)
   - 3-stage curriculum learning (balanced → augmented → realistic)
   - Data augmentation with obfuscation variants
   - Multi-component loss function (BCE + sparsity + counterfactual)

4. **Evaluation System** (`evaluator.py`)
   - Comprehensive testing: classification, localization, speed, robustness
   - Meets strict success criteria for production deployment
   - Benchmark integration for comparison with other models

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install torch transformers numpy scikit-learn matplotlib seaborn tqdm

# Or using uv (recommended)
uv sync
```

### Basic Usage

```python
from amil import create_amil_model, AMILFeatureExtractor
from amil.config import create_default_config

# Create default configuration
config, training_config, eval_config = create_default_config()

# Initialize components
feature_extractor = AMILFeatureExtractor(config)
model = create_amil_model(config, device="cuda")

# For training
from amil.trainer import AMILTrainer
trainer = AMILTrainer(model, feature_extractor, config, training_config)
trainer.train(train_samples)

# save model 
trainer.save_model("trained_amil.pth", push_to_hub=True, repo_name="amil-malicious-package-detection")

# For inference
from amil.model import predict_package
result = predict_package(model, feature_extractor, package_files)
```

### Demo Script

```bash
# Run comprehensive demo showing all capabilities
python amil_demo.py
```

## 📊 Model Architecture

### Multiple Instance Learning Approach

AMIL treats each package as a "bag" of code units (functions/files), where:
- **Bag**: Package containing multiple units
- **Instances**: Individual code units within the package
- **Label**: Package-level malicious/benign classification

### Attention Mechanism

```
Package Units → Feature Extraction → Multi-Head Attention → Classification
     |                |                      |                    |
 [unit1, unit2,   [embed1, embed2,    [attention_weights]   [malicious/benign
  unit3, ...]       embed3, ...]          ↓                   + confidence]
                                    Weighted Aggregation
```

### Loss Function

The model is trained with three complementary loss components:

1. **Binary Cross-Entropy**: Standard classification loss
2. **Sparsity Loss**: Encourages attention to focus on relevant units
3. **Counterfactual Loss**: Ensures top-attended units are truly important

## 🎓 Training Process

### Curriculum Learning (3 Stages)

**Stage A: Balanced Training**
- 5:1 benign:malicious ratio
- Clean, unobfuscated samples
- Establishes basic discrimination

**Stage B: Augmented Training** 
- Adds obfuscated variants (minification, base64, string splitting)
- Builds robustness to common evasion techniques
- Same 5:1 ratio with augmentation

**Stage C: Realistic Training**
- 10:1 benign:malicious ratio (realistic deployment)
- Calibrates confidence scores for production use
- Final robustness validation

### Data Augmentation

- **Minification**: Remove whitespace and comments
- **Base64 Encoding**: Encode string literals
- **String Splitting**: Break strings across lines
- **Identifier Obfuscation**: Rename variables/functions

## 📈 Evaluation Metrics

### Success Criteria

- **Detection**: ROC-AUC ≥ 0.95
- **Precision**: False positive rate < 2%
- **Localization**: IoU ≥ 0.7 (attention overlaps malicious units)
- **Speed**: ≤2s inference per package
- **Robustness**: Maintains performance under obfuscation

### Comprehensive Testing

1. **Classification Metrics**: ROC-AUC, Precision, Recall, F1
2. **Speed Benchmarking**: Inference time across package sizes
3. **Localization Analysis**: Attention weight accuracy
4. **Robustness Testing**: Performance under various obfuscations
5. **Ablation Studies**: Component contribution analysis

## 🔌 Integration

### Benchmark Framework

```python
from amil_benchmark_integration import AMILBenchmarkModel

# Create benchmark-compatible model
amil_model = AMILBenchmarkModel(model_path="trained_amil.pth")

# Use in existing benchmark suite
benchmark_suite.register_model(amil_model)
results = await benchmark_suite.run_evaluation()
```

### CI/CD Integration

```python
# Fast package screening
def screen_package(package_path: str) -> dict:
    result = amil_model.predict_package_from_path(package_path)
    return {
        "verdict": "MALICIOUS" if result.is_malicious else "BENIGN",
        "confidence": result.confidence,
        "suspicious_files": result.top_suspicious_units,
        "inference_time": result.inference_time
    }
```

## 🛠️ Configuration

### Model Configuration

```python
@dataclass
class AMILConfig:
    # Architecture
    unit_embedding_dim: int = 256
    attention_heads: int = 8
    hidden_dim: int = 128
    
    # Training
    max_units_per_package: int = 50
    dropout_rate: float = 0.3
    
    # Features
    api_categories: Dict[str, List[str]] = field(default_factory=...)
    entropy_threshold: float = 6.0
    
    # Curriculum stages
    curriculum_stages: Dict[str, Dict] = field(default_factory=...)
```

### Training Configuration

```python
@dataclass  
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    max_epochs_per_stage: int = 30
    early_stopping_patience: int = 5
    
    # Loss weights
    bce_weight: float = 1.0
    sparsity_weight: float = 0.1
    counterfactual_weight: float = 0.2
```

## 🎯 Use Cases

### 1. CI/CD Pipeline Integration
- Fast package screening during builds
- Block malicious dependencies before deployment
- Generate security reports for development teams

### 2. Registry Scanning
- Continuous monitoring of package repositories  
- Identify newly published malicious packages
- Support registry security policies

### 3. Security Analysis
- Forensic analysis of suspicious packages
- Generate detailed explanations for security teams
- Support incident response workflows

### 4. Research & Benchmarking
- Compare against other detection methods
- Evaluate on new malware datasets
- Support academic research in supply chain security

## 🔍 Interpretability Features

### Attention-based Localization
- Identify which code units contributed most to classification
- Visualize attention weights across package structure
- Support analyst workflows with actionable insights

### Detailed Explanations
```python
# Get detailed prediction with explanations
detailed_result = amil_model.get_detailed_prediction(sample)

print(f"Verdict: {detailed_result['package_verdict']}")
print(f"Top suspicious units: {detailed_result['unit_rankings'][:3]}")
print(f"Attention analysis: {detailed_result['attention_analysis']}")
```

## 📚 API Reference

### Core Classes

- `AMILModel`: Main attention-based MIL classifier
- `AMILFeatureExtractor`: Feature extraction pipeline
- `AMILTrainer`: Training pipeline with curriculum learning
- `AMILEvaluator`: Comprehensive evaluation system
- `AMILBenchmarkModel`: Integration with benchmark framework

### Configuration Classes

- `AMILConfig`: Model and feature configuration
- `TrainingConfig`: Training hyperparameters and settings
- `EvaluationConfig`: Evaluation metrics and thresholds

### Utility Functions

- `create_amil_model()`: Factory function for model creation
- `create_default_config()`: Default configuration generator
- `extract_features_from_code()`: Standalone feature extraction

## 🐛 Troubleshooting

### Common Issues

**CUDA Memory Issues**
```python
# Reduce batch size or max units
config.max_units_per_package = 25
training_config.batch_size = 16
```

**Slow Feature Extraction**
```python
# Use CPU for CodeBERT if GPU memory limited
config.feature_extraction_device = "cpu"
```

**Poor Attention Localization**
```python
# Increase sparsity loss weight
training_config.sparsity_weight = 0.2
```

## 🤝 Contributing

1. Follow existing code style and documentation patterns
2. Add tests for new features
3. Update benchmarks for model changes
4. Ensure compatibility with ICN framework integration

## 📝 License

This project is part of the Zorro malicious package detection framework.