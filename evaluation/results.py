"""
Results analysis and management for the evaluation system.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Unified results analyzer for all evaluation types."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.all_results = self._load_all_results()

    def _load_all_results(self) -> Dict[str, Any]:
        """Load all available results files."""
        results = {}

        # Look for various result files
        result_files = {
            'evaluation_summary': 'evaluation_summary.json',
            'openrouter': 'openrouter/openrouter_results.json',
            'local': 'local/local_results.json',
            'huggingface': 'huggingface/huggingface_results.json',
            'baseline': 'baseline/baseline_results.json'
        }

        for key, filename in result_files.items():
            filepath = self.results_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        results[key] = json.load(f)
                    logger.info(f"📊 Loaded {key} results from {filename}")
                except Exception as e:
                    logger.warning(f"Failed to load {filename}: {e}")

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get high-level summary of all results."""
        summary = {
            'available_result_types': list(self.all_results.keys()),
            'total_evaluations': 0,
            'model_types': set(),
            'best_performers': {}
        }

        # Analyze each result type
        for result_type, data in self.all_results.items():
            if result_type == 'evaluation_summary':
                summary['evaluation_name'] = data.get('evaluation_name')
                summary['duration'] = data.get('duration_seconds')
                summary['total_samples'] = data.get('data_summary', {}).get('samples_count')
            elif 'results' in data:
                # Count successful evaluations
                successful = sum(1 for r in data['results'] if r.get('success', True))
                summary['total_evaluations'] += successful

                # Extract model types
                for result in data['results']:
                    model_name = result.get('model_name', '')
                    if 'openai' in model_name or 'anthropic' in model_name:
                        summary['model_types'].add('llm')
                    elif 'icn' in model_name.lower():
                        summary['model_types'].add('icn')
                    elif 'amil' in model_name.lower():
                        summary['model_types'].add('amil')
                    else:
                        summary['model_types'].add('other')

        summary['model_types'] = list(summary['model_types'])
        return summary

    def compare_across_types(self) -> pd.DataFrame:
        """Compare performance across different model types."""
        all_performance = []

        for result_type, data in self.all_results.items():
            if 'results' in data:
                for result in data['results']:
                    if result.get('success', True):
                        all_performance.append({
                            'model_type': result_type,
                            'model_name': result.get('model_name'),
                            'accuracy': self._extract_accuracy(result),
                            'confidence': result.get('confidence', 0.5),
                            'cost': result.get('cost_usd', 0.0),
                            'time': result.get('inference_time_seconds', 0.0),
                            'prediction': result.get('prediction'),
                            'ground_truth': result.get('ground_truth')
                        })

        if not all_performance:
            return pd.DataFrame()

        df = pd.DataFrame(all_performance)
        return df

    def _extract_accuracy(self, result: Dict[str, Any]) -> float:
        """Extract accuracy from a result record."""
        if 'prediction' in result and 'ground_truth' in result:
            return 1.0 if result['prediction'] == result['ground_truth'] else 0.0
        return result.get('accuracy', 0.5)

    def generate_cross_model_report(self) -> str:
        """Generate report comparing all model types."""
        df = self.compare_across_types()

        if df.empty:
            return "No results available for cross-model comparison."

        report = f"""# Cross-Model Type Comparison

## Performance by Model Type

"""

        # Group by model type
        type_summary = df.groupby('model_type').agg({
            'accuracy': ['mean', 'count'],
            'confidence': 'mean',
            'cost': 'mean',
            'time': 'mean'
        }).round(3)

        type_summary.columns = ['avg_accuracy', 'sample_count', 'avg_confidence', 'avg_cost', 'avg_time']

        report += "| Model Type | Avg Accuracy | Samples | Avg Confidence | Avg Cost | Avg Time |\n"
        report += "|------------|-------------|---------|----------------|----------|----------|\n"

        for model_type, row in type_summary.iterrows():
            report += f"| {model_type} | {row['avg_accuracy']:.3f} | {int(row['sample_count'])} | {row['avg_confidence']:.3f} | ${row['avg_cost']:.4f} | {row['avg_time']:.2f}s |\n"

        # Best performers
        report += f"\n## Best Performers by Type\n\n"

        for model_type in df['model_type'].unique():
            type_df = df[df['model_type'] == model_type]
            best_model = type_df.loc[type_df['accuracy'].idxmax()]
            report += f"**{model_type.title()}**: {best_model['model_name']} (Accuracy: {best_model['accuracy']:.3f})\n"

        return report

    def export_to_csv(self, output_path: Optional[str] = None) -> str:
        """Export all results to CSV format."""
        df = self.compare_across_types()

        if df.empty:
            logger.warning("No data to export to CSV")
            return ""

        if output_path is None:
            output_path = self.results_dir / "all_results.csv"
        else:
            output_path = Path(output_path)

        df.to_csv(output_path, index=False)
        logger.info(f"📊 Exported results to {output_path}")

        return str(output_path)

    def get_cost_analysis(self) -> Dict[str, Any]:
        """Analyze costs across all evaluations."""
        df = self.compare_across_types()

        if df.empty or 'cost' not in df.columns:
            return {'error': 'No cost data available'}

        cost_analysis = {
            'total_cost': df['cost'].sum(),
            'avg_cost_per_prediction': df['cost'].mean(),
            'cost_by_model_type': df.groupby('model_type')['cost'].agg(['sum', 'mean', 'count']).to_dict(),
            'most_expensive': df.loc[df['cost'].idxmax()].to_dict() if len(df) > 0 else None,
            'most_cost_effective': None
        }

        # Calculate cost effectiveness (accuracy per dollar)
        df['cost_effectiveness'] = df['accuracy'] / (df['cost'] + 0.001)  # Avoid division by zero
        if len(df) > 0:
            cost_analysis['most_cost_effective'] = df.loc[df['cost_effectiveness'].idxmax()].to_dict()

        return cost_analysis