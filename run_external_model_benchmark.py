#!/usr/bin/env python3
"""
External Model Multi-Prompt Benchmarking Runner
Tests all OpenRouter models with multiple prompt strategies - excludes local ICN/AMIL models.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
import json

# Add ICN to path
sys.path.append(str(Path(__file__).parent))

from multi_prompt_benchmark import MultiPromptBenchmarkSuite, PromptEffectivenessAnalyzer
from icn.evaluation.openrouter_client import OpenRouterClient
from icn.evaluation.prepare_benchmark_data import BenchmarkDataPreparator


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    Path("logs").mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/external_model_benchmark.log'),
            logging.StreamHandler()
        ]
    )


async def main():
    """Main benchmarking execution."""
    max_samples = 50
    use_existing_data = True
    models = [
        "mistralai/mistral-7b-instruct:free",
        "nvidia/nemotron-nano-9b-v2:free",
        "qwen/qwen3-30b-a3b-thinking-2507",
        "deepseek/deepseek-chat-v3.1:free",
        "openai/gpt-5",
        "openai/gpt-5-mini",
        "openai/gpt-oss-120b:free",
        "openai/gpt-oss-20b:free",
        "mistralai/devstral-small",
        "google/gemma-3n-e2b-it:free",
        "tencent/hunyuan-a13b-instruct:free",
        "mistralai/mistral-small-3.2-24b-instruct:free",
        "agentica-org/deepcoder-14b-preview:free",
        "qwen/qwen3-8b:free",
        "qwen/qwen3-4b:free",
        "meta-llama/llama-3.3-8b-instruct:free",
        "google/gemma-3n-e4b-it:free",
        "anthropic/claude-opus-4",
        "qwen/qwen-2.5-coder-32b-instruct",
        "qwen/qwen-2.5-7b-instruct",
        "google/gemma-3-12b-it",
        "microsoft/phi-4-reasoning-plus",
    ]
    strategies = ["zero_shot", "few_shot", "reasoning", "file_by_file"]
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", None)
    max_concurrent = 2
    output_dir = "external_benchmark_results"
    log_level = "INFO"
    quick_test = False
    dry_run = False
    cost_limit = 25.0

    # Setup logging
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    # Quick test overrides
    if quick_test:
        max_samples = 10
        max_concurrent = 1
        if not models:
            models = ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"]  # Just 2 models
        if not strategies:
            strategies = ["zero_shot", "reasoning"]  # Just 2 strategies

    logger.info("🚀 Starting External Model Multi-Prompt Benchmark")
    logger.info(f"   Max samples: {max_samples}")
    logger.info(f"   Models: {models if models else 'all available'}")
    logger.info(f"   Strategies: {strategies if strategies else 'all compatible'}")
    logger.info(f"   Cost limit: ${cost_limit}")
    logger.info(f"   Output directory: {output_dir}")

    try:
        # Phase 1: Data Preparation
        logger.info("📊 Phase 1: Benchmark Data Preparation")

        data_preparator = BenchmarkDataPreparator()

        # Check for existing data
        benchmark_data_path = Path("data/benchmark_samples/test_samples.json")

        if use_existing_data and benchmark_data_path.exists():
            logger.info("📥 Loading existing benchmark samples...")
            test_samples = data_preparator.load_benchmark_samples(benchmark_data_path)
        else:
            logger.info("📤 Creating fresh benchmark data...")
            train_samples, test_samples = data_preparator.create_benchmark_split_from_icn_data(
                max_samples_per_category=max_samples,
                test_split_ratio=0.3,  # More test data for better evaluation
                force_recompute=not use_existing_data
            )

            # Save for future use
            output_dir = Path("data/benchmark_samples")
            output_dir.mkdir(parents=True, exist_ok=True)
            data_preparator.save_benchmark_samples(test_samples, benchmark_data_path)

        if not test_samples:
            logger.error("❌ No benchmark samples available!")
            return

        logger.info(f"✅ Loaded {len(test_samples)} benchmark samples")

        # Phase 2: Model Setup
        logger.info("🤖 Phase 2: External Model Setup")

        # Check OpenRouter API key
        openrouter_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            logger.error("❌ OpenRouter API key required!")
            logger.error("   Set OPENROUTER_API_KEY environment variable or use --openrouter-api-key")
            return

        # Initialize benchmark suite
        benchmark = MultiPromptBenchmarkSuite(output_dir=output_dir)
        benchmark.load_samples(test_samples)

        # Initialize OpenRouter client
        async with OpenRouterClient(api_key=openrouter_key) as openrouter_client:

            # Register models for multi-prompt testing
            models_to_test = models if models else None
            registered_models = benchmark.register_openrouter_models(
                openrouter_client,
                model_ids=models_to_test
            )

            if registered_models == 0:
                logger.error("❌ No models registered for benchmarking!")
                return

            logger.info(f"✅ Registered {registered_models} models for multi-prompt testing")

            # Estimate costs
            estimated_cost = estimate_benchmark_cost(
                num_models=registered_models,
                num_samples=len(test_samples),
                strategies_per_model=len(benchmark.prompt_strategies),
                openrouter_client=openrouter_client,
                model_ids=models_to_test
            )

            logger.info(f"💰 Estimated cost: ${estimated_cost:.2f}")

            if estimated_cost > cost_limit:
                logger.warning(f"⚠️  Estimated cost (${estimated_cost:.2f}) exceeds limit (${cost_limit})!")
                if not quick_test:
                    response = input("Continue anyway? (y/N): ").strip().lower()
                    if response != 'y':
                        logger.info("Benchmark cancelled by user.")
                        return

            if dry_run:
                logger.info("🔍 Dry run complete - exiting before benchmark execution")
                logger.info(f"Would test {registered_models} models with {len(test_samples)} samples")
                logger.info(f"Estimated cost: ${estimated_cost:.2f}")
                return

            # Phase 3: Multi-Prompt Benchmark Execution
            logger.info("🚀 Phase 3: Multi-Prompt Benchmark Execution")
            logger.info("   This may take a while depending on the number of models and samples...")

            # Run the benchmark
            results_df = await benchmark.run_multi_prompt_benchmark(
                openrouter_client,
                max_concurrent=max_concurrent
            )

            # Phase 4: Analysis and Reporting
            logger.info("📊 Phase 4: Results Analysis and Reporting")

            # Create analyzer
            analyzer = benchmark.analyze_results()

            # Generate comprehensive report
            report = analyzer.generate_comparison_report()

            # Save detailed results
            benchmark.save_multi_prompt_results("detailed_results.json")

            # Save analysis results
            analysis_results = analyzer.analyze_prompt_effectiveness()
            analysis_file = Path(output_dir) / "prompt_effectiveness_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)

            # Save report
            report_file = Path(output_dir) / "multi_prompt_benchmark_report.md"
            with open(report_file, 'w') as f:
                f.write(report)

            # Print summary to console
            print("\n" + "=" * 80)
            print("🎯 EXTERNAL MODEL MULTI-PROMPT BENCHMARK COMPLETE")
            print("=" * 80)
            print(report)
            print("=" * 80)
            print(f"📁 Detailed results: {output_dir}/")
            print(f"📊 Analysis: {analysis_file}")
            print(f"📝 Report: {report_file}")

            # Print top performers
            if 'overall_strategy_ranking' in analysis_results:
                print(f"\n🏆 TOP PROMPT STRATEGIES:")
                for i, strategy in enumerate(analysis_results['overall_strategy_ranking'][:3], 1):
                    print(f"   {i}. {strategy['strategy']}: Accuracy = {strategy['accuracy']:.3f}, Efficiency = {strategy['efficiency_score']:.2f}")

            if 'best_strategies_per_model' in analysis_results:
                print(f"\n🤖 BEST STRATEGY PER MODEL:")
                for model, info in list(analysis_results['best_strategies_per_model'].items())[:5]:
                    print(f"   {model}: {info['strategy']} (Accuracy: {info['accuracy']:.3f})")

            # Print final cost summary
            usage_summary = openrouter_client.get_usage_summary()
            print(f"\n💰 FINAL COST SUMMARY:")
            print(f"   Total cost: ${usage_summary['total_cost_usd']:.2f}")
            print(f"   Total requests: {usage_summary['total_requests']}")
            print(f"   Average latency: {usage_summary['average_latency_seconds']:.2f}s")
            print(f"   Cost per request: ${usage_summary.get('cost_per_request', 0):.4f}")

            print("=" * 80)
            logger.info("🎉 Multi-prompt benchmark completed successfully!")

    except KeyboardInterrupt:
        logger.info("⚠️  Benchmark interrupted by user")

    except Exception as e:
        logger.error(f"💥 Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def estimate_benchmark_cost(num_models: int, num_samples: int, strategies_per_model: int,
                            openrouter_client: OpenRouterClient, model_ids: List[str] = None) -> float:
    """Estimate the cost of running the benchmark."""

    # Average tokens per request (rough estimates)
    avg_prompt_tokens = 800  # Depends on package content size
    avg_completion_tokens = 200  # JSON response
    avg_total_tokens = avg_prompt_tokens + avg_completion_tokens

    total_requests = num_models * num_samples * strategies_per_model

    if model_ids:
        # Calculate weighted average cost
        total_cost = 0.0
        for model_id in model_ids:
            if model_id in openrouter_client.models:
                model_config = openrouter_client.models[model_id]
                model_cost = (avg_total_tokens / 1000) * model_config.cost_per_1k_tokens
                model_requests = num_samples * strategies_per_model
                total_cost += model_cost * model_requests
    else:
        # Use average cost across all models
        avg_cost_per_1k = 0.015  # Average across all models
        cost_per_request = (avg_total_tokens / 1000) * avg_cost_per_1k
        total_cost = cost_per_request * total_requests

    return total_cost


if __name__ == "__main__":
    # Run the benchmark
    asyncio.run(main())
