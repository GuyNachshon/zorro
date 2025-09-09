# 🏆 ICN Comprehensive Benchmark Study

A complete benchmarking framework to compare ICN (Intent Convergence Networks) against state-of-the-art malicious package detection models.

## 📊 **Overview**

This benchmark evaluates ICN's performance against:
- **Actual Security Models**: Endor Labs BERT malicious package classifier
- **Modern LLMs**: GPT o1, Claude 3.5, Gemini 2.0, and top open source models via OpenRouter
- **Traditional Baselines**: Heuristic detection and classical ML approaches

## 🚀 **Quick Start**

### 1. **Setup Environment**

```bash
# Install dependencies
uv sync

# Set up API keys (optional, for LLM comparison)
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

### 2. **Test Framework**

```bash
# Verify everything works
python test_benchmark_framework.py
```

### 3. **Run Quick Benchmark**

```bash
# Quick test with limited samples
python run_icn_benchmark.py --quick-test --include-baselines --dry-run

# Full quick benchmark (requires API key for LLMs)
python run_icn_benchmark.py --quick-test --include-baselines --include-llms
```

### 4. **Full Benchmark Study**

```bash
# Complete benchmark (after ICN model is trained)
python run_icn_benchmark.py \
  --icn-checkpoint "checkpoints/icn-model/best_model.pt" \
  --include-huggingface \
  --include-llms \
  --include-baselines \
  --max-concurrent 3
```

## 🏗️ **Architecture**

### **Core Components**

```
icn/evaluation/
├── openrouter_client.py      # OpenRouter API integration
├── benchmark_framework.py    # Unified model interfaces
├── prepare_benchmark_data.py # Data preparation pipeline  
├── metrics.py               # Comprehensive evaluation metrics
└── README.md               # This documentation
```

### **Model Interfaces**

- **`ICNBenchmarkModel`**: Our ICN model wrapper
- **`HuggingFaceModel`**: Endor Labs BERT and other HF models  
- **`OpenRouterModel`**: LLM models via OpenRouter API
- **`BaselineModel`**: Heuristic and classical ML baselines

## 📈 **Benchmarked Models**

### **Primary Model**
- **ICN with Curriculum Learning**: Our 4-stage trained model

### **Security Baselines**
- **Endor Labs BERT**: `endorlabs/malicious-package-classifier-bert-mal-only`
- **Heuristic Detection**: Pattern-based suspicious code detection
- **Random Baseline**: For statistical comparison

### **LLMs (via OpenRouter)**

#### **Reasoning Models**
- **GPT o1**: Latest OpenAI reasoning model
- **GPT o1-mini**: Faster reasoning variant  
- **Claude 3.5 Sonnet**: Advanced reasoning capabilities

#### **Frontier Models**
- **GPT-4o**: Latest multimodal model
- **Gemini 1.5 Pro**: Google's advanced model
- **Gemini 2.0 Flash**: Latest high-speed model

#### **Open Source Models**
- **Qwen3-Coder-32B**: Latest coding model
- **DeepSeek-Coder-V2**: Advanced code understanding
- **Hermes-3-405B**: Top reasoning model
- **Llama-3.3-70B**: Latest Meta model

## 📊 **Evaluation Metrics**

### **Primary Metrics**
- **F1 Score**: Harmonic mean of precision/recall (main comparison metric)
- **Precision**: True positives / (true positives + false positives)  
- **Recall**: True positives / (true positives + false negatives)
- **ROC-AUC**: Area under ROC curve

### **Secondary Metrics**
- **Inference Time**: Speed per package evaluation
- **API Cost**: Dollar cost for LLM models (via OpenRouter)
- **Success Rate**: Percentage of successful predictions
- **Interpretability**: Quality of model explanations

## 🗂️ **Dataset**

### **Sample Categories**
- **Benign**: Legitimate packages (5,000 samples)
- **Compromised Libraries**: Trojaned packages with malicious units
- **Malicious Intent**: Packages built to be malicious from scratch
- **Total**: ~12,000 packages from npm and PyPI ecosystems

### **Data Split**
- **Training**: 80% for few-shot LLM examples
- **Testing**: 20% for evaluation (balanced across categories)

## 🔧 **Configuration Options**

### **Basic Usage**
```bash
python run_icn_benchmark.py [OPTIONS]
```

### **Key Options**

| Option | Description | Example |
|--------|-------------|---------|
| `--icn-checkpoint` | Path to ICN model | `"checkpoints/icn/best.pt"` |
| `--include-huggingface` | Include HF models | `--include-huggingface` |
| `--include-llms` | Include LLM models | `--include-llms` |
| `--include-baselines` | Include traditional baselines | `--include-baselines` |
| `--max-samples` | Limit samples (testing) | `--max-samples 200` |
| `--quick-test` | Fast test mode | `--quick-test` |
| `--dry-run` | Setup without execution | `--dry-run` |

### **LLM Configuration**
```bash
# Specify models to test
--llm-models openai/gpt-4o anthropic/claude-3.5-sonnet google/gemini-flash-2.0

# Set OpenRouter API key
export OPENROUTER_API_KEY="your_key_here"
```

## 📋 **Output Files**

After running the benchmark, find results in `benchmark_results/`:

```
benchmark_results/
├── benchmark_results_detailed.json    # Raw predictions and metadata
├── metrics_summary.json              # Performance metrics by model  
├── benchmark_report.md               # Human-readable report
└── openrouter_usage.json            # API cost breakdown (if applicable)
```

### **Sample Report Format**

```markdown
# ICN Malicious Package Detection Benchmark Report

## Model Performance Summary

| Model | F1 Score | Precision | Recall | ROC-AUC | Avg Time (s) | Success Rate |
|-------|----------|-----------|---------|---------|--------------|--------------|
| ICN | 0.912 | 0.895 | 0.930 | 0.945 | 0.023 | 1.000 |
| GPT_o1 | 0.887 | 0.901 | 0.874 | 0.925 | 3.245 | 0.985 |
| Endor_Labs_BERT | 0.834 | 0.812 | 0.857 | 0.889 | 0.156 | 0.999 |
| Claude_3.5 | 0.823 | 0.845 | 0.802 | 0.901 | 2.891 | 0.978 |
| Heuristic_Baseline | 0.743 | 0.689 | 0.806 | 0.798 | 0.001 | 1.000 |
```

## 🎯 **Success Criteria**

### **Primary Goals**
- **ICN F1 > 0.85**: Competitive with SOTA models
- **Speed Advantage**: ICN faster than LLM approaches
- **Cost Efficiency**: Better performance/cost ratio than APIs
- **Interpretability**: Superior explanations vs black-box models

### **Research Questions**
1. **How does ICN compare to existing security models?**
2. **Do reasoning models (GPT o1) outperform standard LLMs for malware detection?**
3. **Can open source models match proprietary LLM performance?**
4. **What's the speed/accuracy trade-off across different approaches?**

## 🔬 **Advanced Usage**

### **Custom Model Integration**

Add your own model by implementing the `BaseModel` interface:

```python
from icn.evaluation.benchmark_framework import BaseModel, BenchmarkResult

class MyCustomModel(BaseModel):
    def __init__(self):
        super().__init__("MyModel")
    
    async def predict(self, sample: BenchmarkSample) -> BenchmarkResult:
        # Your prediction logic here
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        return {"model_type": "Custom", ...}

# Register with benchmark
benchmark.register_model(MyCustomModel())
```

### **Custom Evaluation Scenarios**

```python
# Cross-ecosystem evaluation
python run_icn_benchmark.py --train-ecosystem npm --test-ecosystem pypi

# Few-shot LLM evaluation  
python run_icn_benchmark.py --llm-prompt-type few_shot --few-shot-examples 5

# Adversarial robustness testing
python run_icn_benchmark.py --include-adversarial --obfuscation-level high
```

## 🚨 **Troubleshooting**

### **Common Issues**

**1. OpenRouter API Errors**
```bash
# Check API key
echo $OPENROUTER_API_KEY

# Test connection
python -c "from icn.evaluation.openrouter_client import OpenRouterClient; print('OK')"
```

**2. Memory Issues**
```bash
# Reduce concurrent evaluations
--max-concurrent 1

# Limit sample size
--max-samples 100
```

**3. Missing Dependencies**
```bash
# Reinstall dependencies
uv sync --reinstall
```

### **Debug Mode**
```bash
python run_icn_benchmark.py --log-level DEBUG --dry-run
```

## 📚 **Research Integration**

This benchmark framework is designed for research publication. Key features:

- **Reproducible Results**: Deterministic evaluation with seed control
- **Statistical Significance**: Built-in confidence intervals and p-tests
- **Comprehensive Metrics**: Beyond accuracy - speed, cost, interpretability
- **Cross-Ecosystem Validation**: Generalization testing across npm/pypi
- **Error Analysis**: Detailed failure case studies

## 🎉 **Next Steps**

1. **Train ICN Model**: `python train_icn.py`
2. **Get OpenRouter Key**: [openrouter.ai](https://openrouter.ai)
3. **Run Benchmark**: `python run_icn_benchmark.py --quick-test`
4. **Analyze Results**: Review generated reports
5. **Iterate**: Refine ICN model based on benchmark insights

---

**Ready to benchmark ICN against the world's best malware detection models!** 🚀