# Zorro Evaluation Package Guide

## Overview

The Zorro Evaluation Package provides a **unified, YAML-configured system** for running all types of malicious package detection benchmarks. It supports:

- **OpenRouter Models** (GPT-4o, Claude, Gemini, etc.) with **multi-prompt strategies**
- **Local Models** (ICN, AMIL)
- **HuggingFace Models** (Endor Labs BERT, etc.)
- **Baseline Models** (heuristics, random)

All controlled through **simple YAML configuration files**.

## Quick Start

### 1. Create Example Configurations
```bash
python evaluate.py create-configs
```

This creates:
- `evaluation/configs/quick_test.yaml` - Quick 2-model test
- `evaluation/configs/external_comprehensive.yaml` - Full external model evaluation
- `evaluation/configs/local_models.yaml` - Local ICN/AMIL models only

### 2. Set Your API Key
```bash
export OPENROUTER_API_KEY="your_key_here"
```

### 3. Run Quick Test
```bash
python evaluate.py run evaluation/configs/quick_test.yaml
```

### 4. Run Full External Model Evaluation
```bash
python evaluate.py run evaluation/configs/external_comprehensive.yaml
```

## YAML Configuration Structure

### Basic Configuration
```yaml
name: "my_evaluation"
description: "Custom evaluation setup"

# Models to test
models:
  - name: gpt4o
    type: openrouter
    openrouter_id: openai/gpt-4o
    enabled: true

  - name: claude
    type: openrouter
    openrouter_id: anthropic/claude-3.5-sonnet
    enabled: true

# Prompt strategies (for external models)
prompts:
  zero_shot: true      # Direct analysis
  few_shot: true       # With examples
  reasoning: true      # Step-by-step (o1 models only)
  file_by_file: false  # Individual file analysis (expensive)

# Data configuration
data:
  max_samples_per_category: 50
  use_cached_data: true

# Execution settings
execution:
  cost_limit_usd: 25.0
  max_concurrent_requests: 2

# Output settings
output:
  output_directory: "my_results"
  save_visualizations: true
  generate_markdown_report: true
```

## Model Types

### OpenRouter Models (External LLMs)
```yaml
models:
  - name: gpt4o
    type: openrouter
    openrouter_id: openai/gpt-4o

  - name: o1_preview
    type: openrouter
    openrouter_id: openai/o1-preview

  - name: claude
    type: openrouter
    openrouter_id: anthropic/claude-3.5-sonnet

  - name: gemini_pro
    type: openrouter
    openrouter_id: google/gemini-pro-1.5

  - name: qwen_coder
    type: openrouter
    openrouter_id: qwen/qwen-3-coder-32b-instruct
```

### Local Models (Your Trained Models)
```yaml
models:
  - name: icn
    type: icn
    model_path: "checkpoints/icn_model.pth"

  - name: amil
    type: amil
    model_path: "checkpoints/amil_model.pth"
```

### HuggingFace Models
```yaml
models:
  - name: endor_bert
    type: huggingface
    hf_model_id: "endor/malware-bert"
```

### Baseline Models
```yaml
models:
  - name: heuristic_baseline
    type: baseline
    parameters:
      baseline_type: heuristic

  - name: random_baseline
    type: baseline
    parameters:
      baseline_type: random
```

## Multi-Prompt Configuration

For **OpenRouter models**, you can enable multiple prompt strategies:

```yaml
prompts:
  zero_shot: true      # ✅ Fast, works with all models
  few_shot: true       # ✅ Better context, moderate cost
  reasoning: true      # ✅ Only for o1 models, expensive but thorough
  file_by_file: false  # ⚠️  Very expensive, analyzes each file individually

  # Custom token limits
  max_tokens:
    zero_shot: 1000
    few_shot: 1500
    reasoning: 2000
    file_by_file: 800

  # Few-shot configuration
  few_shot_examples_count: 3
```

**Multi-Prompt Behavior:**
- Each OpenRouter model tests ALL enabled prompt strategies
- Results are saved separately: `gpt4o_zero_shot`, `gpt4o_few_shot`, etc.
- Comprehensive comparison shows which prompts work best per model

## CLI Commands

### Run Evaluations
```bash
# Run from config file
python evaluate.py run config.yaml

# With overrides
python evaluate.py run config.yaml --cost-limit 50 --max-samples 100

# Force run despite validation warnings
python evaluate.py run config.yaml --force
```

### Analyze Results
```bash
# Basic summary
python evaluate.py analyze results_dir/ --summary

# Cross-model comparison
python evaluate.py analyze results_dir/ --compare

# Cost analysis
python evaluate.py analyze results_dir/ --costs

# Export to CSV
python evaluate.py analyze results_dir/ --export-csv results.csv
```

### Quick Commands
```bash
# Super quick test
python evaluate.py quick-test --api-key YOUR_KEY --samples 3

# Create example configs
python evaluate.py create-configs
```

## Configuration Examples

### External Models Only (Your Request)
```yaml
name: "external_models_comprehensive"
description: "All external models with multiple prompts - no local models"

models:
  - {name: gpt4o, type: openrouter, openrouter_id: "openai/gpt-4o"}
  - {name: o1_preview, type: openrouter, openrouter_id: "openai/o1-preview"}
  - {name: claude, type: openrouter, openrouter_id: "anthropic/claude-3.5-sonnet"}
  - {name: gemini_pro, type: openrouter, openrouter_id: "google/gemini-pro-1.5"}
  - {name: qwen_coder, type: openrouter, openrouter_id: "qwen/qwen-3-coder-32b-instruct"}
  - {name: deepseek, type: openrouter, openrouter_id: "deepseek/deepseek-coder-v2-instruct"}
  - {name: llama33, type: openrouter, openrouter_id: "meta-llama/llama-3.3-70b-instruct"}

prompts:
  zero_shot: true
  few_shot: true
  reasoning: true
  file_by_file: false  # Too expensive for comprehensive test

data:
  max_samples_per_category: 50

execution:
  cost_limit_usd: 40.0
  max_concurrent_requests: 2

output:
  output_directory: "external_comprehensive_results"
  save_visualizations: true
```

### Budget-Conscious Evaluation
```yaml
name: "budget_external_test"

models:
  - {name: qwen_coder, type: openrouter, openrouter_id: "qwen/qwen-3-coder-32b-instruct"}
  - {name: deepseek, type: openrouter, openrouter_id: "deepseek/deepseek-coder-v2-instruct"}

prompts:
  zero_shot: true
  few_shot: false     # Skip to save costs
  reasoning: false    # Skip to save costs
  file_by_file: false

data:
  max_samples_per_category: 25

execution:
  cost_limit_usd: 5.0
```

### Local Models vs Baselines
```yaml
name: "local_vs_baselines"

models:
  - {name: icn, type: icn, model_path: "checkpoints/icn_model.pth"}
  - {name: amil, type: amil, model_path: "checkpoints/amil_model.pth"}
  - {name: heuristic, type: baseline, parameters: {baseline_type: heuristic}}
  - {name: random, type: baseline, parameters: {baseline_type: random}}

prompts: {}  # Local models don't use prompts

execution:
  cost_limit_usd: 0.0  # No API costs
```

## Output Structure

After running an evaluation, you get:

```
results_directory/
├── evaluation_summary.json     # Complete results
├── evaluation_report.md        # Human-readable report
├── openrouter/                 # OpenRouter model results
│   └── openrouter_results.json
├── local/                      # Local model results
│   └── local_results.json
├── visualizations/             # Charts and plots (if enabled)
│   ├── accuracy_heatmap.png
│   ├── cost_performance_scatter.png
│   └── model_ranking_accuracy.png
└── comprehensive_analysis/     # Statistical analysis (if enabled)
    ├── comprehensive_rankings.json
    ├── best_strategies_per_model.json
    └── comprehensive_analysis_report.md
```

## Cost Management

The system provides **comprehensive cost control**:

### Built-in Cost Estimation
```bash
python evaluate.py run config.yaml
# Output: 💰 Estimated OpenRouter cost: $15.75
# Continues only if under cost_limit_usd
```

### Cost Configuration Options
```yaml
execution:
  cost_limit_usd: 25.0           # Hard limit
  max_concurrent_requests: 2     # Rate limiting
  delay_between_requests: 0.1    # API politeness
  timeout_seconds: 120           # Request timeout
```

### Cost-Saving Tips
- **Start small**: Use `max_samples_per_category: 10` for testing
- **Skip expensive strategies**: Disable `file_by_file` and `reasoning`
- **Use cheaper models**: Qwen, DeepSeek instead of GPT-4o
- **Reduce concurrency**: Lower `max_concurrent_requests` if hitting rate limits

## Advanced Features

### Custom Prompts
```yaml
prompts:
  custom_prompts:
    security_focused: "Analyze this code for security vulnerabilities..."
    performance_focused: "Focus on performance implications..."
```

### Statistical Analysis
```yaml
enable_statistical_testing: true
statistical_significance_level: 0.05
```

### Multiple Output Formats
```yaml
output:
  generate_markdown_report: true   # Human-readable report
  generate_json_summary: true      # Machine-readable results
  generate_csv_export: true        # Spreadsheet format
  save_visualizations: true        # Charts and plots
```

## Example Workflow

### 1. Your Specific Use Case
```bash
# Create comprehensive external model config
python evaluate.py create-configs

# Edit the external_comprehensive.yaml if needed
# Then run full evaluation
python evaluate.py run evaluation/configs/external_comprehensive.yaml

# Analyze results
python evaluate.py analyze external_comprehensive_results/ --summary --compare --costs

# Generate visualizations manually if needed
python benchmark_visualization.py external_comprehensive_results/openrouter/openrouter_results.json
```

### 2. Budget-Conscious Testing
```bash
# Quick test first
python evaluate.py run evaluation/configs/quick_test.yaml

# If results look good, run larger evaluation
python evaluate.py run evaluation/configs/external_comprehensive.yaml --cost-limit 15
```

### 3. Research-Grade Evaluation
```bash
# Comprehensive evaluation with all analysis
python evaluate.py run evaluation/configs/external_comprehensive.yaml --max-samples 100 --cost-limit 75

# Generate additional analysis
python comprehensive_model_comparison.py external_comprehensive_results/openrouter/openrouter_results.json
```

## Key Benefits

✅ **YAML Configuration** - No more editing Python scripts
✅ **Multi-Prompt Support** - Test all strategies automatically
✅ **Cost Control** - Built-in estimation and limits
✅ **All Model Types** - External, local, HuggingFace, baselines
✅ **Rich Analysis** - Statistical testing, visualizations, reports
✅ **Unified Interface** - One command for all evaluation types
✅ **Results Preservation** - All individual results saved, no data loss

This gives you exactly what you requested: **a clean way to benchmark all external models with multiple prompts and comprehensive comparison**, all controlled through simple YAML configuration!