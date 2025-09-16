#!/usr/bin/env python3
"""
Comprehensive Model Comparison and Ranking System
Advanced analysis, statistical testing, and ranking for multi-prompt benchmark results.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Performance metrics for a model-strategy combination."""
    model_name: str
    strategy: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    avg_confidence: float
    avg_cost: float
    avg_time: float
    sample_count: int
    success_rate: float
    
    def __str__(self):
        return f"{self.model_name}_{self.strategy}: F1={self.f1_score:.3f}, Acc={self.accuracy:.3f}, Cost=${self.avg_cost:.4f}"


@dataclass
class ComparisonResult:
    """Result of statistical comparison between two models."""
    model_a: str
    model_b: str
    metric: str
    p_value: float
    is_significant: bool
    effect_size: float
    winner: Optional[str]
    
    def __str__(self):
        significance = "significant" if self.is_significant else "not significant"
        winner_str = f", {self.winner} wins" if self.winner else ""
        return f"{self.model_a} vs {self.model_b} ({self.metric}): p={self.p_value:.4f} ({significance}){winner_str}"


class ComprehensiveModelComparator:
    """Advanced comparison and ranking system for benchmark results."""
    
    def __init__(self, results_path: str, significance_level: float = 0.05):
        self.results_path = Path(results_path)
        self.significance_level = significance_level
        
        # Load and process data
        self.raw_data = self._load_results()
        self.results_df = self._create_dataframe()
        self.performances = self._calculate_performances()
        
        logger.info(f"🔬 Loaded {len(self.results_df)} results for comprehensive analysis")
    
    def _load_results(self) -> Dict[str, Any]:
        """Load benchmark results from JSON file."""
        try:
            with open(self.results_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load results from {self.results_path}: {e}")
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        data = []
        
        for result in self.raw_data.get('results', []):
            if result.get('success', False):
                # Extract base model name
                full_name = result['model_name']
                parts = full_name.split('_')
                base_model = '_'.join(parts[:-1]) if len(parts) > 1 else full_name
                
                data.append({
                    'base_model': base_model,
                    'model_name': full_name,
                    'prompt_strategy': result.get('prompt_strategy', 'unknown'),
                    'sample_id': result['sample_id'],
                    'ground_truth': result['ground_truth'],
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'inference_time': result['inference_time_seconds'],
                    'cost_usd': result.get('cost_usd', 0.0),
                    'correct': result['prediction'] == result['ground_truth'],
                    'true_positive': (result['prediction'] == 1) and (result['ground_truth'] == 1),
                    'true_negative': (result['prediction'] == 0) and (result['ground_truth'] == 0),
                    'false_positive': (result['prediction'] == 1) and (result['ground_truth'] == 0),
                    'false_negative': (result['prediction'] == 0) and (result['ground_truth'] == 1)
                })
        
        return pd.DataFrame(data)
    
    def _calculate_performances(self) -> List[ModelPerformance]:
        """Calculate comprehensive performance metrics for each model-strategy combination."""
        performances = []
        
        for (base_model, strategy), group in self.results_df.groupby(['base_model', 'prompt_strategy']):
            if len(group) < 3:  # Skip combinations with too few samples
                continue
            
            # Basic metrics
            accuracy = group['correct'].mean()
            sample_count = len(group)
            
            # Precision, Recall, F1
            tp = group['true_positive'].sum()
            tn = group['true_negative'].sum()
            fp = group['false_positive'].sum()
            fn = group['false_negative'].sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            performance = ModelPerformance(
                model_name=base_model,
                strategy=strategy,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                avg_confidence=group['confidence'].mean(),
                avg_cost=group['cost_usd'].mean(),
                avg_time=group['inference_time'].mean(),
                sample_count=sample_count,
                success_rate=1.0  # All loaded results are successful
            )
            
            performances.append(performance)
        
        return performances
    
    def statistical_comparison(self, metric: str = 'correct') -> List[ComparisonResult]:
        """Perform statistical comparisons between all model-strategy pairs."""
        comparisons = []
        
        # Get all unique model-strategy combinations
        combinations = [(perf.model_name, perf.strategy) for perf in self.performances]
        
        for i, (model_a, strategy_a) in enumerate(combinations):
            for model_b, strategy_b in combinations[i+1:]:
                # Get data for both combinations
                data_a = self.results_df[
                    (self.results_df['base_model'] == model_a) & 
                    (self.results_df['prompt_strategy'] == strategy_a)
                ][metric].values
                
                data_b = self.results_df[
                    (self.results_df['base_model'] == model_b) & 
                    (self.results_df['prompt_strategy'] == strategy_b)
                ][metric].values
                
                if len(data_a) < 3 or len(data_b) < 3:
                    continue
                
                # Perform statistical test
                if metric == 'correct':
                    # For binary outcomes, use chi-square or Fisher's exact test
                    # Simplified to t-test for now
                    statistic, p_value = stats.ttest_ind(data_a, data_b)
                else:
                    # For continuous variables, use t-test
                    statistic, p_value = stats.ttest_ind(data_a, data_b)
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(((len(data_a) - 1) * np.var(data_a, ddof=1) + 
                                    (len(data_b) - 1) * np.var(data_b, ddof=1)) / 
                                   (len(data_a) + len(data_b) - 2))
                effect_size = (np.mean(data_a) - np.mean(data_b)) / pooled_std if pooled_std > 0 else 0.0
                
                # Determine winner
                winner = None
                if p_value < self.significance_level:
                    if np.mean(data_a) > np.mean(data_b):
                        winner = f"{model_a}_{strategy_a}"
                    else:
                        winner = f"{model_b}_{strategy_b}"
                
                comparison = ComparisonResult(
                    model_a=f"{model_a}_{strategy_a}",
                    model_b=f"{model_b}_{strategy_b}",
                    metric=metric,
                    p_value=p_value,
                    is_significant=p_value < self.significance_level,
                    effect_size=abs(effect_size),
                    winner=winner
                )
                
                comparisons.append(comparison)
        
        return sorted(comparisons, key=lambda x: x.p_value)
    
    def create_comprehensive_ranking(self, weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Create comprehensive ranking considering multiple metrics."""
        if weights is None:
            weights = {
                'f1_score': 0.35,
                'accuracy': 0.25,
                'precision': 0.15,
                'recall': 0.15,
                'cost_efficiency': 0.05,  # 1 / avg_cost
                'speed_efficiency': 0.05   # 1 / avg_time
            }
        
        rankings = []
        
        for perf in self.performances:
            # Calculate efficiency scores
            cost_efficiency = 1.0 / (perf.avg_cost + 0.001)  # Avoid division by zero
            speed_efficiency = 1.0 / (perf.avg_time + 0.001)
            
            # Normalize efficiency scores to 0-1 range
            max_cost_eff = max([1.0 / (p.avg_cost + 0.001) for p in self.performances])
            max_speed_eff = max([1.0 / (p.avg_time + 0.001) for p in self.performances])
            
            normalized_cost_eff = cost_efficiency / max_cost_eff if max_cost_eff > 0 else 0
            normalized_speed_eff = speed_efficiency / max_speed_eff if max_speed_eff > 0 else 0
            
            # Calculate weighted score
            composite_score = (
                weights['f1_score'] * perf.f1_score +
                weights['accuracy'] * perf.accuracy +
                weights['precision'] * perf.precision +
                weights['recall'] * perf.recall +
                weights['cost_efficiency'] * normalized_cost_eff +
                weights['speed_efficiency'] * normalized_speed_eff
            )
            
            ranking_entry = {
                'model': perf.model_name,
                'strategy': perf.strategy,
                'composite_score': composite_score,
                'f1_score': perf.f1_score,
                'accuracy': perf.accuracy,
                'precision': perf.precision,
                'recall': perf.recall,
                'avg_confidence': perf.avg_confidence,
                'avg_cost': perf.avg_cost,
                'avg_time': perf.avg_time,
                'sample_count': perf.sample_count,
                'cost_efficiency': normalized_cost_eff,
                'speed_efficiency': normalized_speed_eff,
                'performance_obj': perf
            }
            
            rankings.append(ranking_entry)
        
        # Sort by composite score
        return sorted(rankings, key=lambda x: x['composite_score'], reverse=True)
    
    def find_best_strategy_per_model(self) -> Dict[str, Dict[str, Any]]:
        """Find the best prompt strategy for each model."""
        best_strategies = {}
        
        for model in self.results_df['base_model'].unique():
            model_perfs = [p for p in self.performances if p.model_name == model]
            
            if not model_perfs:
                continue
            
            # Rank strategies by F1 score, then accuracy
            best_perf = max(model_perfs, key=lambda p: (p.f1_score, p.accuracy))
            
            # Get alternative metrics for comparison
            strategies_comparison = []
            for perf in model_perfs:
                strategies_comparison.append({
                    'strategy': perf.strategy,
                    'f1_score': perf.f1_score,
                    'accuracy': perf.accuracy,
                    'avg_cost': perf.avg_cost,
                    'avg_time': perf.avg_time
                })
            
            best_strategies[model] = {
                'best_strategy': best_perf.strategy,
                'best_f1': best_perf.f1_score,
                'best_accuracy': best_perf.accuracy,
                'best_cost': best_perf.avg_cost,
                'all_strategies': sorted(strategies_comparison, 
                                       key=lambda x: x['f1_score'], reverse=True)
            }
        
        return best_strategies
    
    def cost_effectiveness_analysis(self) -> Dict[str, Any]:
        """Analyze cost-effectiveness of different models and strategies."""
        cost_analysis = {
            'most_cost_effective': [],
            'cost_brackets': {
                'budget': [],      # < $0.01 per prediction
                'moderate': [],    # $0.01 - $0.05 per prediction  
                'premium': []      # > $0.05 per prediction
            },
            'cost_vs_performance': []
        }
        
        for perf in self.performances:
            # Calculate cost-effectiveness score (F1 per dollar spent)
            cost_effectiveness = perf.f1_score / (perf.avg_cost + 0.001)
            
            entry = {
                'model': perf.model_name,
                'strategy': perf.strategy,
                'f1_score': perf.f1_score,
                'avg_cost': perf.avg_cost,
                'cost_effectiveness': cost_effectiveness
            }
            
            cost_analysis['cost_vs_performance'].append(entry)
            
            # Categorize by cost brackets
            if perf.avg_cost < 0.01:
                cost_analysis['cost_brackets']['budget'].append(entry)
            elif perf.avg_cost < 0.05:
                cost_analysis['cost_brackets']['moderate'].append(entry)
            else:
                cost_analysis['cost_brackets']['premium'].append(entry)
        
        # Sort by cost-effectiveness
        cost_analysis['most_cost_effective'] = sorted(
            cost_analysis['cost_vs_performance'], 
            key=lambda x: x['cost_effectiveness'], 
            reverse=True
        )[:10]
        
        # Sort within brackets
        for bracket in cost_analysis['cost_brackets'].values():
            bracket.sort(key=lambda x: x['f1_score'], reverse=True)
        
        return cost_analysis
    
    def generate_comprehensive_report(self) -> str:
        """Generate detailed comprehensive comparison report."""
        rankings = self.create_comprehensive_ranking()
        best_strategies = self.find_best_strategy_per_model()
        cost_analysis = self.cost_effectiveness_analysis()
        statistical_comparisons = self.statistical_comparison('correct')
        
        report = f"""# Comprehensive Model Comparison Report

Generated: {datetime.now().isoformat()}
Total Models Tested: {len(self.results_df['base_model'].unique())}
Total Strategies: {len(self.results_df['prompt_strategy'].unique())}
Total Predictions: {len(self.results_df)}

## Executive Summary

### 🏆 Top 5 Overall Performers (Weighted Composite Score)

| Rank | Model | Strategy | Score | F1 | Accuracy | Avg Cost | Avg Time |
|------|-------|----------|-------|----|---------|---------|---------| 
"""
        
        for i, entry in enumerate(rankings[:5], 1):
            report += f"| {i} | {entry['model']} | {entry['strategy']} | {entry['composite_score']:.3f} | {entry['f1_score']:.3f} | {entry['accuracy']:.3f} | ${entry['avg_cost']:.4f} | {entry['avg_time']:.2f}s |\n"
        
        report += f"\n### 💰 Most Cost-Effective Models\n\n"
        
        for i, entry in enumerate(cost_analysis['most_cost_effective'][:5], 1):
            effectiveness = entry['cost_effectiveness']
            report += f"{i}. **{entry['model']}** ({entry['strategy']}): {effectiveness:.1f} F1 per $1\n"
        
        report += f"\n## Best Strategy Per Model\n\n"
        
        for model, info in best_strategies.items():
            report += f"### {model}\n"
            report += f"- **Best Strategy**: {info['best_strategy']}\n"
            report += f"- **Performance**: F1={info['best_f1']:.3f}, Accuracy={info['best_accuracy']:.3f}\n"
            report += f"- **Cost**: ${info['best_cost']:.4f} per prediction\n\n"
            
            report += "All Strategies Comparison:\n"
            for strat in info['all_strategies'][:3]:  # Top 3
                report += f"  - {strat['strategy']}: F1={strat['f1_score']:.3f}, Cost=${strat['avg_cost']:.4f}\n"
            report += "\n"
        
        report += f"## Cost Bracket Analysis\n\n"
        
        brackets = [('Budget (< $0.01)', 'budget'), ('Moderate ($0.01-$0.05)', 'moderate'), ('Premium (> $0.05)', 'premium')]
        
        for bracket_name, bracket_key in brackets:
            bracket_data = cost_analysis['cost_brackets'][bracket_key]
            if bracket_data:
                report += f"### {bracket_name}\n"
                report += f"Best performers in this bracket:\n"
                for i, entry in enumerate(bracket_data[:3], 1):
                    report += f"{i}. {entry['model']} ({entry['strategy']}): F1={entry['f1_score']:.3f}, Cost=${entry['avg_cost']:.4f}\n"
                report += "\n"
        
        report += f"## Statistical Significance Tests\n\n"
        
        significant_comparisons = [c for c in statistical_comparisons if c.is_significant][:10]
        
        if significant_comparisons:
            report += f"Top {len(significant_comparisons)} statistically significant differences (p < {self.significance_level}):\n\n"
            for comp in significant_comparisons:
                report += f"- **{comp.winner}** significantly outperforms competitor (p={comp.p_value:.4f}, effect size={comp.effect_size:.3f})\n"
        else:
            report += "No statistically significant differences found between model-strategy combinations.\n"
        
        report += f"\n## Detailed Performance Matrix\n\n"
        
        report += "| Model | Strategy | F1 | Precision | Recall | Accuracy | Avg Cost | Avg Time | Samples |\n"
        report += "|-------|----------|----|-----------|---------|---------|---------|---------|---------|\n"
        
        for entry in rankings:
            perf = entry['performance_obj']
            report += f"| {perf.model_name} | {perf.strategy} | {perf.f1_score:.3f} | {perf.precision:.3f} | {perf.recall:.3f} | {perf.accuracy:.3f} | ${perf.avg_cost:.4f} | {perf.avg_time:.2f}s | {perf.sample_count} |\n"
        
        report += f"\n## Recommendations\n\n"
        
        # Generate recommendations based on analysis
        best_overall = rankings[0]
        most_cost_effective = cost_analysis['most_cost_effective'][0]
        
        report += f"### For Maximum Performance\n"
        report += f"Use **{best_overall['model']}** with **{best_overall['strategy']}** strategy.\n"
        report += f"- Expected F1 Score: {best_overall['f1_score']:.3f}\n"
        report += f"- Expected Cost: ${best_overall['avg_cost']:.4f} per prediction\n\n"
        
        report += f"### For Cost-Effectiveness\n"
        report += f"Use **{most_cost_effective['model']}** with **{most_cost_effective['strategy']}** strategy.\n"
        report += f"- Expected F1 Score: {most_cost_effective['f1_score']:.3f}\n"
        report += f"- Expected Cost: ${most_cost_effective['avg_cost']:.4f} per prediction\n"
        report += f"- Cost-Effectiveness: {most_cost_effective['cost_effectiveness']:.1f} F1 per $1\n\n"
        
        report += f"### Strategy Recommendations\n"
        strategy_performance = {}
        for perf in self.performances:
            if perf.strategy not in strategy_performance:
                strategy_performance[perf.strategy] = []
            strategy_performance[perf.strategy].append(perf.f1_score)
        
        avg_strategy_performance = {k: np.mean(v) for k, v in strategy_performance.items()}
        best_strategy_overall = max(avg_strategy_performance, key=avg_strategy_performance.get)
        
        report += f"- **{best_strategy_overall}** shows the best average performance across all models\n"
        report += f"- Consider testing **reasoning** prompts with o1 models for complex analysis\n"
        report += f"- Use **zero_shot** prompts for quick, cost-effective screening\n"
        
        return report
    
    def save_detailed_analysis(self, output_dir: str):
        """Save all analysis results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive ranking
        rankings = self.create_comprehensive_ranking()
        rankings_data = [
            {k: v for k, v in entry.items() if k != 'performance_obj'} 
            for entry in rankings
        ]
        
        with open(output_path / "comprehensive_rankings.json", 'w') as f:
            json.dump(rankings_data, f, indent=2, default=str)
        
        # Save best strategies
        best_strategies = self.find_best_strategy_per_model()
        with open(output_path / "best_strategies_per_model.json", 'w') as f:
            json.dump(best_strategies, f, indent=2, default=str)
        
        # Save cost analysis
        cost_analysis = self.cost_effectiveness_analysis()
        with open(output_path / "cost_effectiveness_analysis.json", 'w') as f:
            json.dump(cost_analysis, f, indent=2, default=str)
        
        # Save statistical comparisons
        statistical_comparisons = self.statistical_comparison('correct')
        comparisons_data = [
            {
                'model_a': c.model_a,
                'model_b': c.model_b,
                'metric': c.metric,
                'p_value': c.p_value,
                'is_significant': c.is_significant,
                'effect_size': c.effect_size,
                'winner': c.winner
            }
            for c in statistical_comparisons
        ]
        
        with open(output_path / "statistical_comparisons.json", 'w') as f:
            json.dump(comparisons_data, f, indent=2, default=str)
        
        # Save comprehensive report
        report = self.generate_comprehensive_report()
        with open(output_path / "comprehensive_analysis_report.md", 'w') as f:
            f.write(report)
        
        logger.info(f"📊 Comprehensive analysis saved to {output_path}/")


def main():
    """Example usage of the comprehensive comparison system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive model comparison and ranking")
    parser.add_argument("results_file", help="Path to benchmark results JSON file")
    parser.add_argument("--output-dir", default="comprehensive_analysis", help="Output directory")
    parser.add_argument("--significance-level", type=float, default=0.05, help="Statistical significance level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Create comprehensive comparator
        comparator = ComprehensiveModelComparator(args.results_file, args.significance_level)
        
        # Generate comprehensive analysis
        comparator.save_detailed_analysis(args.output_dir)
        
        # Print summary
        rankings = comparator.create_comprehensive_ranking()
        print("\n🏆 Top 5 Model-Strategy Combinations:")
        print("="*80)
        for i, entry in enumerate(rankings[:5], 1):
            print(f"{i}. {entry['model']} ({entry['strategy']})")
            print(f"   Composite Score: {entry['composite_score']:.3f}")
            print(f"   F1: {entry['f1_score']:.3f}, Accuracy: {entry['accuracy']:.3f}")
            print(f"   Cost: ${entry['avg_cost']:.4f}, Time: {entry['avg_time']:.2f}s")
            print()
        
        print(f"📊 Complete analysis saved to: {args.output_dir}/")
        print("📝 See comprehensive_analysis_report.md for detailed findings")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()