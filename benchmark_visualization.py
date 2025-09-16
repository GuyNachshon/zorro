#!/usr/bin/env python3
"""
Visualization and Analysis Tools for Multi-Prompt Benchmarking
Creates charts, heatmaps, and detailed analysis reports.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class BenchmarkVisualizer:
    """Creates visualizations for multi-prompt benchmark results."""
    
    def __init__(self, results_path: str, output_dir: str = "benchmark_visualizations"):
        self.results_path = Path(results_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load results
        self.results_data = self._load_results()
        self.results_df = self._create_dataframe()
        
        logger.info(f"📊 Loaded {len(self.results_df)} benchmark results for visualization")
    
    def _load_results(self) -> Dict[str, Any]:
        """Load benchmark results from JSON file."""
        try:
            with open(self.results_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load results from {self.results_path}: {e}")
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        data = []
        
        for result in self.results_data.get('results', []):
            if result.get('success', False):
                # Extract base model name (remove strategy suffix)
                full_name = result['model_name']
                parts = full_name.split('_')
                base_model = '_'.join(parts[:-1]) if len(parts) > 1 else full_name
                
                data.append({
                    'base_model': base_model,
                    'model_name': full_name,
                    'prompt_strategy': result.get('prompt_strategy', 'unknown'),
                    'prompt_type': result.get('prompt_type', 'unknown'),
                    'granularity': result.get('granularity', 'package'),
                    'sample_id': result['sample_id'],
                    'ground_truth': result['ground_truth'],
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'inference_time': result['inference_time_seconds'],
                    'cost_usd': result.get('cost_usd', 0.0),
                    'correct': result['prediction'] == result['ground_truth']
                })
        
        return pd.DataFrame(data)
    
    def create_accuracy_heatmap(self, save_path: Optional[str] = None) -> str:
        """Create heatmap showing accuracy for each model-strategy combination."""
        if self.results_df.empty:
            logger.warning("No data available for accuracy heatmap")
            return ""
        
        # Calculate accuracy matrix
        accuracy_matrix = self.results_df.groupby(['base_model', 'prompt_strategy'])['correct'].mean().unstack(fill_value=0)
        
        # Create the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            accuracy_matrix, 
            annot=True, 
            fmt='.3f', 
            cmap='RdYlGn', 
            center=0.5,
            square=True,
            cbar_kws={'label': 'Accuracy'}
        )
        
        plt.title('Model vs Prompt Strategy Accuracy Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Prompt Strategy', fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "accuracy_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Accuracy heatmap saved to {save_path}")
        return str(save_path)
    
    def create_cost_performance_scatter(self, save_path: Optional[str] = None) -> str:
        """Create scatter plot showing cost vs performance trade-offs."""
        if self.results_df.empty:
            logger.warning("No data available for cost-performance scatter plot")
            return ""
        
        # Calculate metrics per model-strategy combination
        metrics = self.results_df.groupby(['model_name', 'prompt_strategy']).agg({
            'correct': 'mean',  # Accuracy
            'cost_usd': 'mean',  # Average cost per prediction
            'inference_time': 'mean',  # Average time
            'base_model': 'first'  # For coloring
        }).reset_index()
        
        metrics.columns = ['model_name', 'strategy', 'accuracy', 'avg_cost', 'avg_time', 'base_model']
        
        # Create scatter plot
        plt.figure(figsize=(14, 8))
        
        # Create scatter plot with different colors for each base model
        unique_models = metrics['base_model'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_models)))
        
        for i, model in enumerate(unique_models):
            model_data = metrics[metrics['base_model'] == model]
            plt.scatter(
                model_data['avg_cost'], 
                model_data['accuracy'],
                c=[colors[i]], 
                label=model,
                s=100,  # Size of points
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
            
            # Add text labels for each point
            for _, row in model_data.iterrows():
                plt.annotate(
                    row['strategy'], 
                    (row['avg_cost'], row['accuracy']),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.8
                )
        
        plt.xlabel('Average Cost per Prediction (USD)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Cost vs Performance Trade-offs by Model and Strategy', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "cost_performance_scatter.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Cost-performance scatter plot saved to {save_path}")
        return str(save_path)
    
    def create_strategy_comparison_bar_chart(self, save_path: Optional[str] = None) -> str:
        """Create bar chart comparing prompt strategies across all models."""
        if self.results_df.empty:
            logger.warning("No data available for strategy comparison")
            return ""
        
        # Calculate metrics by strategy
        strategy_metrics = self.results_df.groupby('prompt_strategy').agg({
            'correct': ['mean', 'count'],
            'confidence': 'mean',
            'inference_time': 'mean',
            'cost_usd': 'mean'
        }).round(4)
        
        # Flatten column names
        strategy_metrics.columns = ['accuracy', 'sample_count', 'avg_confidence', 'avg_time', 'avg_cost']
        strategy_metrics = strategy_metrics.reset_index()
        
        # Create subplot with multiple metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Prompt Strategy Comparison Across All Models', fontsize=16, fontweight='bold')
        
        # Accuracy
        axes[0, 0].bar(strategy_metrics['prompt_strategy'], strategy_metrics['accuracy'], 
                       color='skyblue', edgecolor='navy', alpha=0.7)
        axes[0, 0].set_title('Accuracy by Strategy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(strategy_metrics['accuracy']):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Average Cost
        axes[0, 1].bar(strategy_metrics['prompt_strategy'], strategy_metrics['avg_cost'], 
                       color='lightcoral', edgecolor='darkred', alpha=0.7)
        axes[0, 1].set_title('Average Cost per Prediction')
        axes[0, 1].set_ylabel('Cost (USD)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(strategy_metrics['avg_cost']):
            axes[0, 1].text(i, v + max(strategy_metrics['avg_cost']) * 0.01, f'${v:.4f}', 
                           ha='center', va='bottom', fontweight='bold')
        
        # Confidence
        axes[1, 0].bar(strategy_metrics['prompt_strategy'], strategy_metrics['avg_confidence'], 
                       color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        axes[1, 0].set_title('Average Confidence')
        axes[1, 0].set_ylabel('Confidence')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(strategy_metrics['avg_confidence']):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Inference Time
        axes[1, 1].bar(strategy_metrics['prompt_strategy'], strategy_metrics['avg_time'], 
                       color='gold', edgecolor='orange', alpha=0.7)
        axes[1, 1].set_title('Average Inference Time')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(strategy_metrics['avg_time']):
            axes[1, 1].text(i, v + max(strategy_metrics['avg_time']) * 0.01, f'{v:.2f}s', 
                           ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "strategy_comparison_bars.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Strategy comparison bar chart saved to {save_path}")
        return str(save_path)
    
    def create_model_ranking_chart(self, metric: str = 'accuracy', save_path: Optional[str] = None) -> str:
        """Create horizontal bar chart ranking models by specified metric."""
        if self.results_df.empty:
            logger.warning("No data available for model ranking")
            return ""
        
        # Calculate ranking metric for each model (best strategy per model)
        model_best = self.results_df.groupby(['base_model', 'prompt_strategy']).agg({
            'correct': 'mean',
            'confidence': 'mean',
            'cost_usd': 'mean',
            'inference_time': 'mean'
        }).reset_index()
        
        # Find best strategy per model based on accuracy
        model_rankings = model_best.loc[model_best.groupby('base_model')['correct'].idxmax()].copy()
        model_rankings = model_rankings.sort_values('correct', ascending=True)  # Ascending for horizontal bar
        
        # Create horizontal bar chart
        plt.figure(figsize=(12, 8))
        
        metric_map = {
            'accuracy': 'correct',
            'confidence': 'confidence', 
            'cost': 'cost_usd',
            'time': 'inference_time'
        }
        
        y_col = metric_map.get(metric, 'correct')
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_rankings)))
        
        bars = plt.barh(range(len(model_rankings)), model_rankings[y_col], color=colors, alpha=0.8)
        
        # Customize the chart
        plt.yticks(range(len(model_rankings)), 
                  [f"{model}\n({strategy})" for model, strategy in 
                   zip(model_rankings['base_model'], model_rankings['prompt_strategy'])],
                  fontsize=10)
        
        plt.xlabel(f'{metric.capitalize()}', fontsize=12)
        plt.title(f'Model Ranking by Best {metric.capitalize()} (with optimal strategy)', 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, model_rankings[y_col])):
            if metric == 'cost':
                label = f'${value:.4f}'
            elif metric == 'time':
                label = f'{value:.2f}s'
            else:
                label = f'{value:.3f}'
            
            plt.text(bar.get_width() + max(model_rankings[y_col]) * 0.01, 
                    bar.get_y() + bar.get_height()/2, 
                    label, ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"model_ranking_{metric}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Model ranking chart ({metric}) saved to {save_path}")
        return str(save_path)
    
    def create_confidence_distribution_plots(self, save_path: Optional[str] = None) -> str:
        """Create distribution plots for confidence scores by correctness."""
        if self.results_df.empty:
            logger.warning("No data available for confidence distribution")
            return ""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall confidence distribution
        correct_conf = self.results_df[self.results_df['correct']]['confidence']
        incorrect_conf = self.results_df[~self.results_df['correct']]['confidence']
        
        axes[0].hist(correct_conf, bins=20, alpha=0.7, label='Correct Predictions', color='green', density=True)
        axes[0].hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect Predictions', color='red', density=True)
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Confidence Distribution by Correctness')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Confidence by strategy
        strategies = self.results_df['prompt_strategy'].unique()
        for strategy in strategies:
            strategy_data = self.results_df[self.results_df['prompt_strategy'] == strategy]
            axes[1].hist(strategy_data['confidence'], bins=15, alpha=0.6, label=strategy, density=True)
        
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Confidence Distribution by Strategy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "confidence_distributions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Confidence distribution plots saved to {save_path}")
        return str(save_path)
    
    def generate_all_visualizations(self) -> Dict[str, str]:
        """Generate all visualizations and return paths."""
        logger.info("🎨 Generating all visualizations...")
        
        plots = {}
        
        try:
            plots['accuracy_heatmap'] = self.create_accuracy_heatmap()
            plots['cost_performance_scatter'] = self.create_cost_performance_scatter()
            plots['strategy_comparison'] = self.create_strategy_comparison_bar_chart()
            plots['model_ranking_accuracy'] = self.create_model_ranking_chart('accuracy')
            plots['model_ranking_cost'] = self.create_model_ranking_chart('cost')
            plots['confidence_distributions'] = self.create_confidence_distribution_plots()
            
            logger.info(f"✅ Generated {len(plots)} visualizations in {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to generate some visualizations: {e}")
        
        return plots
    
    def create_summary_statistics(self) -> Dict[str, Any]:
        """Create comprehensive summary statistics."""
        if self.results_df.empty:
            return {"error": "No data available"}
        
        stats = {
            "overview": {
                "total_predictions": len(self.results_df),
                "unique_models": len(self.results_df['base_model'].unique()),
                "unique_strategies": len(self.results_df['prompt_strategy'].unique()),
                "overall_accuracy": self.results_df['correct'].mean(),
                "total_cost": self.results_df['cost_usd'].sum(),
                "avg_inference_time": self.results_df['inference_time'].mean()
            },
            
            "best_performers": {
                "highest_accuracy": self.results_df.loc[self.results_df.groupby(['base_model', 'prompt_strategy'])['correct'].mean().idxmax()],
                "most_cost_effective": self.results_df.loc[(self.results_df['correct'] / (self.results_df['cost_usd'] + 0.001)).idxmax()],
                "fastest": self.results_df.loc[self.results_df['inference_time'].idxmin()]
            },
            
            "strategy_comparison": self.results_df.groupby('prompt_strategy').agg({
                'correct': ['mean', 'count'],
                'confidence': 'mean',
                'cost_usd': 'mean',
                'inference_time': 'mean'
            }).round(4).to_dict(),
            
            "model_comparison": self.results_df.groupby('base_model').agg({
                'correct': ['mean', 'count'],
                'confidence': 'mean',
                'cost_usd': 'sum',
                'inference_time': 'mean'
            }).round(4).to_dict()
        }
        
        return stats


def main():
    """Example usage of the visualization tools."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate visualizations for benchmark results")
    parser.add_argument("results_file", help="Path to benchmark results JSON file")
    parser.add_argument("--output-dir", default="visualizations", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Create visualizer
        visualizer = BenchmarkVisualizer(args.results_file, args.output_dir)
        
        # Generate all plots
        plots = visualizer.generate_all_visualizations()
        
        # Create summary statistics
        stats = visualizer.create_summary_statistics()
        
        # Save statistics
        stats_file = Path(args.output_dir) / "summary_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print("\n🎯 Visualization Complete!")
        print("="*50)
        print(f"📊 Generated {len(plots)} plots in {args.output_dir}/")
        print(f"📈 Summary statistics: {stats_file}")
        print("\nGenerated plots:")
        for name, path in plots.items():
            print(f"  - {name}: {path}")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise


if __name__ == "__main__":
    main()