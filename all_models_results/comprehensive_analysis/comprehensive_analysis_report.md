# Comprehensive Model Comparison Report

Generated: 2025-09-17T14:29:34.228638
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
