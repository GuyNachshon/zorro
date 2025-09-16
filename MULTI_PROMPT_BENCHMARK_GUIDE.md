# Multi-Prompt External Model Benchmarking Guide

## Overview

This system allows you to benchmark **all external models (OpenRouter LLMs) with multiple prompt strategies** without including your local ICN/AMIL models. It provides comprehensive comparison, analysis, and visualization tools.

## Quick Start

### 1. Set Up API Key
```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

### 2. Run Quick Test
```bash
# Quick test with 2 models, 2 strategies, 10 samples
python run_external_model_benchmark.py --quick-test
```

### 3. Full Benchmark
```bash
# Full benchmark with all models and strategies
python run_external_model_benchmark.py --max-samples 100 --cost-limit 50.0
```

## Available Models

The system automatically tests these OpenRouter models:

### Premium Models
- `openai/gpt-4o` - Latest GPT-4o
- `openai/o1-preview` - GPT o1 reasoning model  
- `openai/o1-mini` - GPT o1 mini reasoning
- `anthropic/claude-3.5-sonnet` - Claude 3.5 Sonnet

### Google Models
- `google/gemini-pro-1.5` - Gemini 1.5 Pro
- `google/gemini-flash-2.0` - Gemini 2.0 Flash

### Open Source Models
- `qwen/qwen-3-coder-32b-instruct` - Qwen 3 Coder
- `nousresearch/hermes-3-llama-3.1-405b` - Hermes 3 Llama
- `deepseek/deepseek-coder-v2-instruct` - DeepSeek Coder V2
- `meta-llama/llama-3.3-70b-instruct` - Llama 3.3

## Prompt Strategies

### 1. Zero-Shot (`zero_shot`)
- Direct analysis without examples
- Fast and cost-effective
- Works with all models

### 2. Few-Shot (`few_shot`)
- Analysis with 2-3 examples
- Better context understanding
- Moderate cost increase

### 3. Reasoning (`reasoning`)
- Step-by-step analysis
- **Only for o1 models**
- Higher cost but better reasoning

### 4. File-by-File (`file_by_file`)
- Individual file analysis + aggregation
- Most thorough but expensive
- Works with all models

## Command Line Options

### Basic Usage
```bash
python run_external_model_benchmark.py [OPTIONS]
```

### Key Options
- `--max-samples N` - Limit samples per category (default: 50)
- `--models MODEL1 MODEL2` - Test specific models only
- `--strategies STRAT1 STRAT2` - Test specific strategies only
- `--cost-limit AMOUNT` - Maximum cost in USD (default: $25)
- `--max-concurrent N` - Concurrent requests (default: 2)
- `--quick-test` - Fast test with minimal samples
- `--dry-run` - Setup without running benchmark

### Examples
```bash
# Test only GPT-4o and Claude with zero-shot and reasoning
python run_external_model_benchmark.py \
    --models openai/gpt-4o anthropic/claude-3.5-sonnet \
    --strategies zero_shot reasoning \
    --max-samples 25

# Test all open source models only
python run_external_model_benchmark.py \
    --models qwen/qwen-3-coder-32b-instruct deepseek/deepseek-coder-v2-instruct meta-llama/llama-3.3-70b-instruct \
    --cost-limit 15.0

# Dry run to estimate costs
python run_external_model_benchmark.py --dry-run --max-samples 100
```

## Output and Analysis

The benchmark produces several output files in your specified directory:

### Core Results
- `detailed_results.json` - Raw benchmark results
- `prompt_effectiveness_analysis.json` - Prompt strategy analysis
- `multi_prompt_benchmark_report.md` - Human-readable report

### Visualizations (Optional)
```bash
# Generate all visualizations
python benchmark_visualization.py detailed_results.json --output-dir visualizations/

# Creates:
# - accuracy_heatmap.png - Model vs strategy performance matrix
# - cost_performance_scatter.png - Cost vs accuracy trade-offs
# - strategy_comparison_bars.png - Strategy comparison across models
# - model_ranking_accuracy.png - Model ranking by best accuracy
# - confidence_distributions.png - Confidence score distributions
```

### Comprehensive Analysis (Optional)
```bash
# Advanced statistical analysis and rankings
python comprehensive_model_comparison.py detailed_results.json --output-dir analysis/

# Creates:
# - comprehensive_rankings.json - Weighted composite scores
# - best_strategies_per_model.json - Optimal strategy per model
# - cost_effectiveness_analysis.json - Cost-effectiveness rankings
# - statistical_comparisons.json - Statistical significance tests
# - comprehensive_analysis_report.md - Detailed analysis report
```

## Understanding Results

### Key Metrics
- **F1 Score** - Primary performance metric (harmonic mean of precision/recall)
- **Accuracy** - Overall correctness
- **Cost per Prediction** - API cost in USD
- **Inference Time** - Speed in seconds
- **Cost Effectiveness** - F1 score per dollar spent

### Report Sections

1. **Top Performers** - Best model-strategy combinations by composite score
2. **Cost-Effective Models** - Best value for money 
3. **Best Strategy Per Model** - Optimal prompt for each model
4. **Cost Bracket Analysis** - Performance within budget/moderate/premium tiers
5. **Statistical Significance** - Robust comparison with p-values
6. **Recommendations** - Actionable guidance

## Cost Management

### Estimation
The system estimates costs before running:
```
💰 Estimated cost: $15.75
⚠️  Estimated cost ($15.75) exceeds limit ($10.00)!
Continue anyway? (y/N):
```

### Cost Factors
- **Model Type**: o1 models cost ~4x more than standard models
- **Prompt Strategy**: Reasoning prompts use more tokens
- **Sample Count**: Linear scaling with number of samples
- **File-by-File**: Analyzes multiple files per package (expensive)

### Budget Guidelines
- **$5-10**: Quick test (~20 samples, 2-3 models, zero-shot only)
- **$15-25**: Standard benchmark (~50 samples, 4-6 models, multiple strategies)  
- **$35-50**: Comprehensive (~100 samples, all models, all strategies)

## Best Practices

### For Quick Evaluation
```bash
python run_external_model_benchmark.py \
    --quick-test \
    --models openai/gpt-4o anthropic/claude-3.5-sonnet \
    --strategies zero_shot
```

### For Production Decision
```bash
python run_external_model_benchmark.py \
    --max-samples 75 \
    --cost-limit 30.0 \
    --max-concurrent 2
```

### For Research/Publication
```bash
python run_external_model_benchmark.py \
    --max-samples 200 \
    --cost-limit 100.0 \
    --use-existing-data

# Then generate comprehensive analysis
python comprehensive_model_comparison.py detailed_results.json
```

## Troubleshooting

### Common Issues

**API Rate Limits**
- Reduce `--max-concurrent` to 1
- The system includes automatic delays

**High Costs**  
- Use `--dry-run` to estimate first
- Reduce `--max-samples`
- Exclude expensive strategies (reasoning, file-by-file)
- Test fewer models initially

**No Results**
- Check `OPENROUTER_API_KEY` is set
- Verify internet connection
- Check logs in `logs/external_model_benchmark.log`

**Out of Memory**
- This shouldn't happen (no local models)
- Check disk space for results files

### Getting Help

1. Check logs: `logs/external_model_benchmark.log`
2. Run with `--log-level DEBUG` for more details
3. Use `--dry-run` to test setup without costs

## Integration with Existing Benchmark

This system is designed to complement (not replace) your existing ICN benchmark system:

- **ICN Benchmark**: Tests your trained models (ICN, AMIL) vs baselines
- **Multi-Prompt Benchmark**: Tests external models with different prompts
- **Combined Analysis**: Compare your models against best external approaches

You can run both systems and combine results for complete analysis.

## Example Workflow

```bash
# 1. Quick test to verify setup
python run_external_model_benchmark.py --quick-test

# 2. Full benchmark
python run_external_model_benchmark.py --max-samples 100 --cost-limit 40.0

# 3. Generate visualizations  
python benchmark_visualization.py external_benchmark_results/detailed_results.json

# 4. Generate comprehensive analysis
python comprehensive_model_comparison.py external_benchmark_results/detailed_results.json

# 5. Review results
ls external_benchmark_results/
# - multi_prompt_benchmark_report.md (start here)
# - comprehensive_analysis_report.md (detailed analysis)  
# - visualizations/ (charts and graphs)
```

This gives you complete insights into which external models and prompt strategies work best for malicious package detection, helping you make informed decisions about model selection and deployment strategies.