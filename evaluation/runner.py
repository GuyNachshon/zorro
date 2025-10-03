"""
Unified evaluation runner that orchestrates all benchmark types based on configuration.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import existing components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from .config import EvaluationConfig, ModelConfig
from .model_service import get_model_service, shutdown_model_service
from .service_aware_model import create_service_aware_model
from multi_prompt_benchmark import MultiPromptBenchmarkSuite
from benchmark_visualization import BenchmarkVisualizer
from comprehensive_model_comparison import ComprehensiveModelComparator
from icn.evaluation.openrouter_client import OpenRouterClient
from icn.evaluation.benchmark_framework import BenchmarkSuite, ICNBenchmarkModel, HuggingFaceModel, BaselineModel
from icn.evaluation.prepare_benchmark_data import BenchmarkDataPreparator

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Unified runner for all types of evaluations."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results = {}

        # Create output directory
        self.output_dir = Path(config.output.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🎯 Initialized evaluation: {config.name}")
        logger.info(f"   Output directory: {self.output_dir}")

    async def run_evaluation(self) -> Dict[str, Any]:
        """Run the complete evaluation based on configuration."""
        start_time = datetime.now()

        try:
            logger.info(f"🚀 Starting evaluation: {self.config.name}")

            # Phase 1: Data preparation
            logger.info("📊 Phase 1: Data Preparation")
            test_samples = await self._prepare_data()

            # Phase 2: Model setup and evaluation
            logger.info("🤖 Phase 2: Model Evaluation")
            evaluation_results = await self._run_models(test_samples)

            # Phase 3: Analysis and reporting
            logger.info("📊 Phase 3: Analysis and Reporting")
            analysis_results = await self._generate_analysis(evaluation_results)

            # Phase 4: Visualization (if enabled)
            if self.config.output.save_visualizations:
                logger.info("🎨 Phase 4: Visualization Generation")
                await self._generate_visualizations()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            final_results = {
                'evaluation_name': self.config.name,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'config': self.config.to_dict(),
                'data_summary': {
                    'samples_count': len(test_samples),
                    'categories': list(set([s.sample_type for s in test_samples]))
                },
                'evaluation_results': evaluation_results,
                'analysis_results': analysis_results
            }

            # Save final results
            if self.config.output.save_raw_results:
                self._save_results(final_results)

            logger.info(f"✅ Evaluation completed in {duration:.1f} seconds")
            return final_results

        except Exception as e:
            logger.error(f"💥 Evaluation failed: {e}")
            raise
        finally:
            # Cleanup: shutdown model service if it was used
            try:
                await shutdown_model_service()
                logger.info("🛑 Model service shutdown")
            except:
                pass  # Service might not have been initialized

    async def _prepare_data(self):
        """Prepare benchmark data based on configuration."""
        data_preparator = BenchmarkDataPreparator()

        if self.config.data.data_path and Path(self.config.data.data_path).exists():
            logger.info("📥 Loading existing benchmark data")
            return data_preparator.load_benchmark_samples(Path(self.config.data.data_path))
        else:
            logger.info("📤 Creating fresh benchmark data")
            train_samples, test_samples = data_preparator.create_benchmark_split_from_icn_data(
                max_samples_per_category=self.config.data.max_samples_per_category,
                test_split_ratio=self.config.data.test_split_ratio,
                force_recompute=not self.config.data.use_cached_data
            )

            # Save for future use if requested
            if self.config.data.use_cached_data:
                data_path = self.output_dir / "benchmark_data.json"
                data_preparator.save_benchmark_samples(test_samples, data_path)
                logger.info(f"💾 Cached benchmark data to {data_path}")

            return test_samples

    async def _run_models(self, test_samples):
        """Run evaluation for all configured models."""
        enabled_models = [m for m in self.config.models if m.enabled]

        # Separate models by type
        openrouter_models = [m for m in enabled_models if m.type == "openrouter"]
        local_models = [m for m in enabled_models if m.type in ["icn", "amil"]]
        hf_models = [m for m in enabled_models if m.type == "huggingface"]
        baseline_models = [m for m in enabled_models if m.type == "baseline"]

        results = {}

        # Run OpenRouter models with multi-prompt
        if openrouter_models and self._has_prompt_strategies():
            logger.info(f"🌐 Running {len(openrouter_models)} OpenRouter models with multi-prompt")
            openrouter_results = await self._run_openrouter_models(openrouter_models, test_samples)
            results['openrouter'] = openrouter_results

        # Run local models
        if local_models:
            logger.info(f"🏠 Running {len(local_models)} local models")
            local_results = await self._run_local_models(local_models, test_samples)
            results['local'] = local_results

        # Run HuggingFace models
        if hf_models:
            logger.info(f"🤗 Running {len(hf_models)} HuggingFace models")
            hf_results = await self._run_huggingface_models(hf_models, test_samples)
            results['huggingface'] = hf_results

        # Run baseline models
        if baseline_models:
            logger.info(f"📏 Running {len(baseline_models)} baseline models")
            baseline_results = await self._run_baseline_models(baseline_models, test_samples)
            results['baseline'] = baseline_results

        return results

    def _has_prompt_strategies(self) -> bool:
        """Check if any prompt strategies are enabled."""
        return any([
            self.config.prompts.zero_shot,
            self.config.prompts.few_shot,
            self.config.prompts.reasoning,
            self.config.prompts.file_by_file
        ])

    async def _run_openrouter_models(self, models: List[ModelConfig], test_samples):
        """Run OpenRouter models with multi-prompt evaluation."""
        if not self.config.api_keys.get('openrouter'):
            logger.error("OpenRouter API key required for OpenRouter models")
            return {'error': 'Missing OpenRouter API key'}

        # Create multi-prompt benchmark suite
        benchmark = MultiPromptBenchmarkSuite(output_dir=str(self.output_dir / "openrouter"))
        benchmark.load_samples(test_samples)

        # Filter prompt strategies based on config
        enabled_strategies = []
        for strategy in benchmark.prompt_strategies:
            if strategy.name == "zero_shot" and self.config.prompts.zero_shot:
                enabled_strategies.append(strategy)
            elif strategy.name == "few_shot" and self.config.prompts.few_shot:
                enabled_strategies.append(strategy)
            elif strategy.name == "reasoning" and self.config.prompts.reasoning:
                enabled_strategies.append(strategy)
            elif strategy.name == "file_by_file" and self.config.prompts.file_by_file:
                enabled_strategies.append(strategy)

        benchmark.prompt_strategies = enabled_strategies

        async with OpenRouterClient(api_key=self.config.api_keys['openrouter']) as openrouter_client:
            # Set custom prompts if provided
            if self.config.prompts.custom_prompts:
                from icn.evaluation.openrouter_client import MaliciousPackagePrompts
                MaliciousPackagePrompts.set_custom_prompts(self.config.prompts.custom_prompts)
                logger.info(f"🎨 Using {len(self.config.prompts.custom_prompts)} custom prompts")

            # Register models
            model_ids = [m.openrouter_id for m in models if m.openrouter_id]
            registered = benchmark.register_openrouter_models(openrouter_client, model_ids)

            if registered == 0:
                return {'error': 'No OpenRouter models could be registered'}

            # Estimate cost
            estimated_cost = self._estimate_openrouter_cost(
                len(models), len(test_samples), len(enabled_strategies)
            )

            logger.info(f"💰 Estimated OpenRouter cost: ${estimated_cost:.2f}")

            if estimated_cost > self.config.execution.cost_limit_usd:
                logger.warning(f"⚠️ Estimated cost exceeds limit!")
                # In production, you might want to abort or ask for confirmation

            # Run benchmark
            results_df = await benchmark.run_multi_prompt_benchmark(
                openrouter_client,
                max_concurrent=self.config.execution.max_concurrent_requests
            )

            # Save results
            benchmark.save_multi_prompt_results("openrouter_results.json")

            return {
                'model_count': registered,
                'results_df': results_df,
                'total_cost': openrouter_client.total_cost,
                'total_requests': openrouter_client.total_requests,
                'benchmark_suite': benchmark
            }

    async def _run_local_models(self, models: List[ModelConfig], test_samples):
        """Run local ICN/AMIL/CPG/NeoBERT models."""
        # First check if we need to train any models that don't exist
        trained_models = []

        for model_config in models:
            model_path = model_config.model_path or f"checkpoints/{model_config.name}/{model_config.name}_model.pth"

            if not Path(model_path).exists():
                logger.info(f"Model checkpoint not found for {model_config.name}, training new model...")

                try:
                    if model_config.type == "icn":
                        # ICN training should already be available via train_icn.py
                        logger.warning(f"ICN training integration not implemented for {model_config.name}")

                    elif model_config.type == "amil":
                        # Import and train AMIL model
                        from train_amil import train_amil_model
                        save_dir = f"checkpoints/{model_config.name}"
                        results = train_amil_model(
                            save_dir=save_dir,
                            **model_config.parameters
                        )
                        model_path = results.get('best_model_path', f"{save_dir}/amil_model.pth")
                        logger.info(f"✅ Trained AMIL model saved to {model_path}")

                    elif model_config.type == "cpg":
                        # Import and train CPG model
                        from train_cpg import train_cpg_model
                        save_dir = f"checkpoints/{model_config.name}"
                        results = train_cpg_model(
                            save_dir=save_dir,
                            **model_config.parameters
                        )
                        model_path = results.get('best_model_path', f"{save_dir}/cpg_model.pth")
                        logger.info(f"✅ Trained CPG model saved to {model_path}")

                    elif model_config.type == "neobert":
                        # Import and train NeoBERT model
                        from train_neobert import train_neobert_model
                        save_dir = f"checkpoints/{model_config.name}"
                        results = train_neobert_model(
                            save_dir=save_dir,
                            **model_config.parameters
                        )
                        model_path = results.get('best_model_path', f"{save_dir}/neobert_model.pth")
                        logger.info(f"✅ Trained NeoBERT model saved to {model_path}")

                    trained_models.append((model_config, model_path))

                except Exception as e:
                    logger.error(f"Failed to train {model_config.name}: {e}")
                    continue
            else:
                trained_models.append((model_config, model_path))

        # Create standard benchmark suite
        benchmark = BenchmarkSuite(output_dir=str(self.output_dir / "local"))
        benchmark.load_samples(test_samples)

        # Register local models
        for model_config, model_path in trained_models:
            if model_config.type == "icn":
                model = ICNBenchmarkModel(model_path=model_path)
                benchmark.register_model(model)
            elif model_config.type in ["amil", "cpg", "neobert"]:
                # For now, log that these aren't integrated with benchmark framework yet
                logger.warning(f"{model_config.type.upper()} model trained but benchmark integration not implemented: {model_config.name}")

        if not benchmark.models:
            return {
                'error': 'No local models could be registered for benchmarking',
                'trained_models': [config.name for config, _ in trained_models],
                'note': 'Models were trained successfully but benchmark integration is only available for ICN models currently'
            }

        # Run benchmark
        results_df = await benchmark.run_benchmark(
            max_concurrent=self.config.execution.max_concurrent_requests
        )

        benchmark.save_results("local_results.json")

        return {
            'model_count': len(benchmark.models),
            'results_df': results_df,
            'benchmark_suite': benchmark,
            'trained_models': [config.name for config, _ in trained_models]
        }

    async def _run_huggingface_models(self, models: List[ModelConfig], test_samples):
        """Run HuggingFace models using ModelService."""
        benchmark = BenchmarkSuite(output_dir=str(self.output_dir / "huggingface"))
        benchmark.load_samples(test_samples)

        # Get model service and pre-load models
        model_service = await get_model_service()

        logger.info("🔄 Pre-loading HuggingFace models in service...")
        load_results = await model_service.load_models_from_config(models)

        successful_models = [name for name, success in load_results.items() if success]
        if not successful_models:
            return {'error': 'No HuggingFace models could be loaded in service'}

        logger.info(f"✅ Loaded {len(successful_models)} models in service: {successful_models}")

        # Register service-aware models
        for model_config in models:
            if model_config.name in successful_models:
                service_model = create_service_aware_model(model_config)
                benchmark.register_model(service_model)

        if not benchmark.models:
            return {'error': 'No HuggingFace models could be registered'}

        # Run benchmark (models will use the service)
        results_df = await benchmark.run_benchmark(
            max_concurrent=self.config.execution.max_concurrent_requests
        )

        benchmark.save_results("huggingface_results.json")

        # Get service status for debugging
        service_status = model_service.get_service_status()

        return {
            'model_count': len(benchmark.models),
            'results_df': results_df,
            'benchmark_suite': benchmark,
            'service_status': service_status,
            'load_results': load_results
        }

    async def _run_baseline_models(self, models: List[ModelConfig], test_samples):
        """Run baseline models."""
        benchmark = BenchmarkSuite(output_dir=str(self.output_dir / "baseline"))
        benchmark.load_samples(test_samples)

        # Register baseline models
        for model_config in models:
            baseline_type = model_config.parameters.get('baseline_type', 'heuristic')
            model = BaselineModel(model_config.name, baseline_type)
            benchmark.register_model(model)

        if not benchmark.models:
            return {'error': 'No baseline models could be registered'}

        # Run benchmark
        results_df = await benchmark.run_benchmark(
            max_concurrent=self.config.execution.max_concurrent_requests
        )

        benchmark.save_results("baseline_results.json")

        return {
            'model_count': len(benchmark.models),
            'results_df': results_df,
            'benchmark_suite': benchmark
        }

    def _estimate_openrouter_cost(self, num_models: int, num_samples: int, num_strategies: int) -> float:
        """Estimate OpenRouter API costs."""
        avg_tokens_per_request = 1000
        avg_cost_per_1k_tokens = 0.015
        total_requests = num_models * num_samples * num_strategies
        return (total_requests * avg_tokens_per_request / 1000) * avg_cost_per_1k_tokens

    async def _generate_analysis(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis of results."""
        analysis = {}

        # Analyze OpenRouter results if available
        if 'openrouter' in evaluation_results and 'benchmark_suite' in evaluation_results['openrouter']:
            benchmark_suite = evaluation_results['openrouter']['benchmark_suite']
            analyzer = benchmark_suite.analyze_results()

            analysis['openrouter'] = {
                'prompt_effectiveness': analyzer.analyze_prompt_effectiveness(),
                'comparison_report': analyzer.generate_comparison_report()
            }

            # Generate comprehensive comparison if requested
            if self.config.enable_statistical_testing:
                # Save openrouter results for comprehensive analysis
                openrouter_results_file = self.output_dir / "openrouter" / "openrouter_results.json"
                if openrouter_results_file.exists():
                    comparator = ComprehensiveModelComparator(
                        str(openrouter_results_file),
                        self.config.statistical_significance_level
                    )
                    comparator.save_detailed_analysis(str(self.output_dir / "comprehensive_analysis"))
                    analysis['comprehensive'] = comparator.generate_comprehensive_report()

        # Analyze other model types similarly
        for model_type in ['local', 'huggingface', 'baseline']:
            if model_type in evaluation_results and 'benchmark_suite' in evaluation_results[model_type]:
                benchmark_suite = evaluation_results[model_type]['benchmark_suite']
                metrics = benchmark_suite.compute_metrics()
                report = benchmark_suite.generate_report()

                analysis[model_type] = {
                    'metrics': metrics,
                    'report': report
                }

        return analysis

    async def _generate_visualizations(self):
        """Generate visualizations if enabled and data is available."""
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Look for results files to visualize
        results_files = []
        for results_file in self.output_dir.rglob("*_results.json"):
            if results_file.stat().st_size > 0:  # Non-empty file
                results_files.append(results_file)

        for results_file in results_files:
            try:
                visualizer = BenchmarkVisualizer(
                    str(results_file),
                    str(viz_dir / results_file.stem)
                )
                plots = visualizer.generate_all_visualizations()
                logger.info(f"📊 Generated visualizations for {results_file.name}")
            except Exception as e:
                logger.warning(f"Failed to generate visualizations for {results_file.name}: {e}")

    def _save_results(self, results: Dict[str, Any]):
        """Save final results in requested formats."""

        # JSON summary (always saved if save_raw_results is True)
        if self.config.output.generate_json_summary:
            json_file = self.output_dir / "evaluation_summary.json"
            import json
            with open(json_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"💾 Saved JSON summary: {json_file}")

        # Markdown report
        if self.config.output.generate_markdown_report:
            md_file = self.output_dir / "evaluation_report.md"
            report = self._generate_markdown_report(results)
            with open(md_file, 'w') as f:
                f.write(report)
            logger.info(f"📝 Saved Markdown report: {md_file}")

        # CSV export (if requested)
        if self.config.output.generate_csv_export:
            # This would export results to CSV format
            logger.info("📊 CSV export requested but not yet implemented")

    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive markdown report."""
        report = f"""# Evaluation Report: {self.config.name}

**Generated:** {datetime.now().isoformat()}
**Duration:** {results.get('duration_seconds', 0):.1f} seconds
**Description:** {self.config.description}

## Configuration Summary

- **Models:** {len([m for m in self.config.models if m.enabled])} enabled
- **Data Samples:** {results.get('data_summary', {}).get('samples_count', 'N/A')}
- **Output Directory:** `{self.config.output.output_directory}`

### Enabled Models
"""

        for model in self.config.models:
            if model.enabled:
                report += f"- **{model.name}** ({model.type})\n"

        report += "\n### Prompt Strategies\n"

        strategies = []
        if self.config.prompts.zero_shot:
            strategies.append("zero_shot")
        if self.config.prompts.few_shot:
            strategies.append("few_shot")
        if self.config.prompts.reasoning:
            strategies.append("reasoning")
        if self.config.prompts.file_by_file:
            strategies.append("file_by_file")

        if strategies:
            for strategy in strategies:
                report += f"- {strategy}\n"
        else:
            report += "- None (local models only)\n"

        report += "\n## Results Summary\n\n"

        # Add analysis results if available
        if 'analysis_results' in results:
            analysis = results['analysis_results']

            if 'openrouter' in analysis and 'comparison_report' in analysis['openrouter']:
                report += "### OpenRouter Models Analysis\n\n"
                report += analysis['openrouter']['comparison_report']
                report += "\n"

            if 'comprehensive' in analysis:
                report += "### Comprehensive Statistical Analysis\n\n"
                report += analysis['comprehensive']
                report += "\n"

        report += f"\n## Files Generated\n\n"
        report += f"- `evaluation_summary.json` - Complete results in JSON format\n"
        report += f"- `evaluation_report.md` - This report\n"

        if self.config.output.save_visualizations:
            report += f"- `visualizations/` - Generated charts and plots\n"

        report += f"\n---\n*Generated by Zorro Evaluation Package v1.0.0*"

        return report


# Convenience functions for easy usage
async def run_evaluation_from_config(config_path: str) -> Dict[str, Any]:
    """Run evaluation from YAML config file."""
    from .config import load_config

    config = load_config(config_path)
    runner = EvaluationRunner(config)
    return await runner.run_evaluation()


async def run_quick_test(api_key: Optional[str] = None) -> Dict[str, Any]:
    """Run a quick test evaluation."""
    from .config import EvaluationConfig, ModelConfig, PromptConfig, DataConfig, ExecutionConfig, OutputConfig

    config = EvaluationConfig(
        name="quick_test",
        description="Quick test with minimal models and samples",
        models=[
            ModelConfig(name="gpt4o", type="openrouter", openrouter_id="openai/gpt-4o"),
            ModelConfig(name="claude", type="openrouter", openrouter_id="anthropic/claude-3.5-sonnet")
        ],
        prompts=PromptConfig(zero_shot=True, few_shot=False, reasoning=False, file_by_file=False),
        data=DataConfig(max_samples_per_category=5),
        execution=ExecutionConfig(cost_limit_usd=2.0, max_concurrent_requests=1),
        output=OutputConfig(output_directory="quick_test_results"),
        api_keys={'openrouter': api_key} if api_key else {}
    )

    runner = EvaluationRunner(config)
    return await runner.run_evaluation()