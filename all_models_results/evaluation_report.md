# Evaluation Report: all_models_comprehensive

**Generated:** 2025-09-17T14:29:38.494548
**Duration:** 707.7 seconds
**Description:** Comprehensive evaluation of ALL model types - external, local, and baselines

## Configuration Summary

- **Models:** 25 enabled
- **Data Samples:** 26
- **Output Directory:** `all_models_results`

### Enabled Models
- **gpt4o** (openrouter)
- **claude** (openrouter)
- **qwen_coder** (openrouter)
- **qwen_coder_3** (openrouter)
- **qwen_3_8b** (openrouter)
- **qwen_3_4b** (openrouter)
- **qwen_3_0.6b** (openrouter)
- **qwen_3_1.7b** (openrouter)
- **qwen3_next_80b_a3b_instruct** (openrouter)
- **qwen3_next_80b_a3b_thinking** (openrouter)
- **llama_3.3_8b_instruct** (openrouter)
- **gemma_3n_e4b_it** (openrouter)
- **gemma_3n_e2b_it** (openrouter)
- **devstral_small_2505** (openrouter)
- **mistral_small_32b_24b_it** (openrouter)
- **mistral_small_1.1** (openrouter)
- **hunyuan_a13b_instruct** (openrouter)
- **gpt_oss_20b** (openrouter)
- **qwen3_30b_a3b_thinking_2507** (openrouter)
- **seed_oss_36b_instruct** (openrouter)
- **nemotron_nano_9b_v2** (openrouter)
- **ernie_4.5_21b_a3b** (openrouter)
- **endorlabs_malicious_classifier** (huggingface)
- **heuristic_baseline** (baseline)
- **random_baseline** (baseline)

### Prompt Strategies
- zero_shot
- few_shot
- reasoning

## Results Summary

### OpenRouter Models Analysis

# Multi-Prompt Benchmarking Analysis Report

Generated: 2025-09-17T14:29:34.214462

## Overall Strategy Ranking

| Strategy | Accuracy | Avg Confidence | Avg Cost ($) | Efficiency Score | Samples |
|----------|----------|----------------|--------------|------------------|---------|
| few_shot | 0.915 | 0.932 | $0.0971 | 9.33 | 47 |
| zero_shot | 0.833 | 0.908 | $0.0817 | 10.08 | 48 |

## Best Strategy Per Model

**openai**: few_shot (Accuracy: 0.962, Cost: $2.1900)
**anthropic**: few_shot (Accuracy: 0.857, Cost: $2.3727)

## Strategy Performance Matrix

| Model | p | n |
|-------|---|---|
| o | N/A | N/A |
| a | N/A | N/A |

### Comprehensive Statistical Analysis

# Comprehensive Model Comparison Report

Generated: 2025-09-17T14:29:34.233540
Total Models Tested: 4
Total Strategies: 2
Total Predictions: 95

## Executive Summary

### 🏆 Top 5 Overall Performers (Weighted Composite Score)

| Rank | Model | Strategy | Score | F1 | Accuracy | Avg Cost | Avg Time |
|------|-------|----------|-------|----|---------|---------|---------| 
| 1 | openai_gpt-4o_few | few_shot | 0.934 | 0.933 | 0.962 | $0.0842 | 7.25s |
| 2 | openai_gpt-4o_zero | zero_shot | 0.770 | 0.667 | 0.846 | $0.0698 | 6.44s |
| 3 | anthropic_claude-3.5-sonnet_few | few_shot | 0.760 | 0.727 | 0.857 | $0.1130 | 13.28s |
| 4 | anthropic_claude-3.5-sonnet_zero | zero_shot | 0.728 | 0.667 | 0.818 | $0.0958 | 11.19s |

### 💰 Most Cost-Effective Models

1. **openai_gpt-4o_few** (few_shot): 11.0 F1 per $1
2. **openai_gpt-4o_zero** (zero_shot): 9.4 F1 per $1
3. **anthropic_claude-3.5-sonnet_zero** (zero_shot): 6.9 F1 per $1
4. **anthropic_claude-3.5-sonnet_few** (few_shot): 6.4 F1 per $1

## Best Strategy Per Model

### openai_gpt-4o_zero
- **Best Strategy**: zero_shot
- **Performance**: F1=0.667, Accuracy=0.846
- **Cost**: $0.0698 per prediction

All Strategies Comparison:
  - zero_shot: F1=0.667, Cost=$0.0698

### openai_gpt-4o_few
- **Best Strategy**: few_shot
- **Performance**: F1=0.933, Accuracy=0.962
- **Cost**: $0.0842 per prediction

All Strategies Comparison:
  - few_shot: F1=0.933, Cost=$0.0842

### anthropic_claude-3.5-sonnet_zero
- **Best Strategy**: zero_shot
- **Performance**: F1=0.667, Accuracy=0.818
- **Cost**: $0.0958 per prediction

All Strategies Comparison:
  - zero_shot: F1=0.667, Cost=$0.0958

### anthropic_claude-3.5-sonnet_few
- **Best Strategy**: few_shot
- **Performance**: F1=0.727, Accuracy=0.857
- **Cost**: $0.1130 per prediction

All Strategies Comparison:
  - few_shot: F1=0.727, Cost=$0.1130

## Cost Bracket Analysis

### Premium (> $0.05)
Best performers in this bracket:
1. openai_gpt-4o_few (few_shot): F1=0.933, Cost=$0.0842
2. anthropic_claude-3.5-sonnet_few (few_shot): F1=0.727, Cost=$0.1130
3. anthropic_claude-3.5-sonnet_zero (zero_shot): F1=0.667, Cost=$0.0958

## Statistical Significance Tests

No statistically significant differences found between model-strategy combinations.

## Detailed Performance Matrix

| Model | Strategy | F1 | Precision | Recall | Accuracy | Avg Cost | Avg Time | Samples |
|-------|----------|----|-----------|---------|---------|---------|---------|---------|
| openai_gpt-4o_few | few_shot | 0.933 | 1.000 | 0.875 | 0.962 | $0.0842 | 7.25s | 26 |
| openai_gpt-4o_zero | zero_shot | 0.667 | 1.000 | 0.500 | 0.846 | $0.0698 | 6.44s | 26 |
| anthropic_claude-3.5-sonnet_few | few_shot | 0.727 | 1.000 | 0.571 | 0.857 | $0.1130 | 13.28s | 21 |
| anthropic_claude-3.5-sonnet_zero | zero_shot | 0.667 | 1.000 | 0.500 | 0.818 | $0.0958 | 11.19s | 22 |

## Recommendations

### For Maximum Performance
Use **openai_gpt-4o_few** with **few_shot** strategy.
- Expected F1 Score: 0.933
- Expected Cost: $0.0842 per prediction

### For Cost-Effectiveness
Use **openai_gpt-4o_few** with **few_shot** strategy.
- Expected F1 Score: 0.933
- Expected Cost: $0.0842 per prediction
- Cost-Effectiveness: 11.0 F1 per $1

### Strategy Recommendations
- **few_shot** shows the best average performance across all models
- Consider testing **reasoning** prompts with o1 models for complex analysis
- Use **zero_shot** prompts for quick, cost-effective screening


## Files Generated

- `evaluation_summary.json` - Complete results in JSON format
- `evaluation_report.md` - This report
- `visualizations/` - Generated charts and plots

---
*Generated by Zorro Evaluation Package v1.0.0*