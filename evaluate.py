#!/usr/bin/env python3
"""
Unified evaluation CLI for the Zorro malicious package detection framework.
Supports YAML-based configuration and multiple evaluation types.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from evaluation import load_config, EvaluationRunner, ResultsAnalyzer
from evaluation.config import create_example_configs


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    Path("logs").mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/evaluation.log'),
            logging.StreamHandler()
        ]
    )


async def run_evaluation(args):
    """Run evaluation from configuration file."""
    try:
        # Load configuration
        config = load_config(args.config)

        # Override config with CLI arguments if provided
        if args.cost_limit:
            config.execution.cost_limit_usd = args.cost_limit
        if args.max_samples:
            config.data.max_samples_per_category = args.max_samples
        if args.concurrent:
            config.execution.max_concurrent_requests = args.concurrent
        if args.output_dir:
            config.output.output_directory = args.output_dir

        # Validate configuration
        issues = config.validate()
        if issues:
            print("⚠️  Configuration issues found:")
            for issue in issues:
                print(f"   - {issue}")

            if not args.force:
                response = input("Continue anyway? (y/N): ").strip().lower()
                if response != 'y':
                    print("Evaluation cancelled.")
                    return

        # Run evaluation
        runner = EvaluationRunner(config)
        results = await runner.run_evaluation()

        # Print summary
        print("\n" + "="*80)
        print("🎯 EVALUATION COMPLETE")
        print("="*80)
        print(f"📊 Evaluation: {results['evaluation_name']}")
        print(f"⏱️  Duration: {results['duration_seconds']:.1f} seconds")
        print(f"📁 Results: {config.output.output_directory}/")

        if 'evaluation_results' in results:
            eval_results = results['evaluation_results']

            if 'openrouter' in eval_results and 'total_cost' in eval_results['openrouter']:
                print(f"💰 Total Cost: ${eval_results['openrouter']['total_cost']:.2f}")

            for model_type, result in eval_results.items():
                if 'model_count' in result:
                    print(f"🤖 {model_type.title()}: {result['model_count']} models tested")

        print("="*80)

    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        print(f"💥 Evaluation failed: {e}")
        if args.debug:
            raise


def analyze_results(args):
    """Analyze existing results."""
    try:
        analyzer = ResultsAnalyzer(args.results_dir)

        if args.summary:
            summary = analyzer.get_summary()
            print("\n📊 Results Summary:")
            print(f"   Evaluation: {summary.get('evaluation_name', 'Unknown')}")
            print(f"   Duration: {summary.get('duration', 0):.1f} seconds")
            print(f"   Total Samples: {summary.get('total_samples', 'N/A')}")
            print(f"   Model Types: {', '.join(summary.get('model_types', []))}")
            print(f"   Total Evaluations: {summary.get('total_evaluations', 0)}")

        if args.compare:
            report = analyzer.generate_cross_model_report()
            print("\n" + "="*50)
            print("📈 CROSS-MODEL COMPARISON")
            print("="*50)
            print(report)

        if args.costs:
            cost_analysis = analyzer.get_cost_analysis()
            if 'error' not in cost_analysis:
                print("\n💰 Cost Analysis:")
                print(f"   Total Cost: ${cost_analysis['total_cost']:.2f}")
                print(f"   Avg Cost per Prediction: ${cost_analysis['avg_cost_per_prediction']:.4f}")

                if cost_analysis.get('most_cost_effective'):
                    best = cost_analysis['most_cost_effective']
                    print(f"   Most Cost-Effective: {best['model_name']} (${best['cost']:.4f})")

        if args.export_csv:
            csv_path = analyzer.export_to_csv(args.export_csv)
            if csv_path:
                print(f"📊 Exported results to: {csv_path}")

    except Exception as e:
        print(f"💥 Analysis failed: {e}")
        if args.debug:
            raise


def create_configs(args):
    """Create example configuration files."""
    try:
        create_example_configs()
        print("✅ Example configuration files created in evaluation/configs/")
        print("   - quick_test.yaml")
        print("   - external_comprehensive.yaml")
        print("   - local_models.yaml")
        print("\nTo run an evaluation:")
        print("   python evaluate.py run evaluation/configs/quick_test.yaml")

    except Exception as e:
        print(f"💥 Failed to create configs: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Zorro Unified Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create example configurations
  python evaluate.py create-configs

  # Run quick test
  python evaluate.py run evaluation/configs/quick_test.yaml

  # Run with overrides
  python evaluate.py run config.yaml --cost-limit 50 --max-samples 100

  # Analyze existing results
  python evaluate.py analyze results/ --summary --compare --costs

  # Export results to CSV
  python evaluate.py analyze results/ --export-csv all_results.csv
        """
    )

    # Global options
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run evaluation
    run_parser = subparsers.add_parser("run", help="Run evaluation from config file")
    run_parser.add_argument("config", help="Path to YAML configuration file")
    run_parser.add_argument("--cost-limit", type=float, help="Override cost limit")
    run_parser.add_argument("--max-samples", type=int, help="Override max samples per category")
    run_parser.add_argument("--concurrent", type=int, help="Override concurrent requests")
    run_parser.add_argument("--output-dir", help="Override output directory")
    run_parser.add_argument("--force", action="store_true", help="Force run despite validation issues")

    # Analyze results
    analyze_parser = subparsers.add_parser("analyze", help="Analyze existing results")
    analyze_parser.add_argument("results_dir", help="Path to results directory")
    analyze_parser.add_argument("--summary", action="store_true", help="Show results summary")
    analyze_parser.add_argument("--compare", action="store_true", help="Compare across model types")
    analyze_parser.add_argument("--costs", action="store_true", help="Show cost analysis")
    analyze_parser.add_argument("--export-csv", help="Export results to CSV file")

    # Create example configs
    config_parser = subparsers.add_parser("create-configs", help="Create example configuration files")

    # Quick commands
    quick_parser = subparsers.add_parser("quick-test", help="Run quick test with minimal setup")
    quick_parser.add_argument("--api-key", help="OpenRouter API key")
    quick_parser.add_argument("--samples", type=int, default=5, help="Samples per category")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "run":
            asyncio.run(run_evaluation(args))
        elif args.command == "analyze":
            analyze_results(args)
        elif args.command == "create-configs":
            create_configs(args)
        elif args.command == "quick-test":
            asyncio.run(run_quick_test(args))

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"💥 Command failed: {e}")
        if args.debug:
            raise


async def run_quick_test(args):
    """Run quick test evaluation."""
    from evaluation.runner import run_quick_test

    print("🚀 Running quick test evaluation...")

    try:
        results = await run_quick_test(args.api_key)

        print("\n✅ Quick test completed!")
        print(f"   Duration: {results['duration_seconds']:.1f} seconds")
        print(f"   Results: {results.get('config', {}).get('output', {}).get('output_directory', 'quick_test_results')}/")

    except Exception as e:
        print(f"💥 Quick test failed: {e}")
        raise


if __name__ == "__main__":
    main()