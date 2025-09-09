#!/usr/bin/env python3
"""
ICN Comprehensive Benchmark Study
Compares ICN against SOTA models for malicious package detection.
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path
import sys

# Add ICN to path
sys.path.append(str(Path(__file__).parent))

from icn.evaluation.benchmark_framework import (
    BenchmarkSuite, ICNBenchmarkModel, HuggingFaceModel, 
    OpenRouterModel, BaselineModel
)
from icn.evaluation.prepare_benchmark_data import BenchmarkDataPreparator
from icn.evaluation.openrouter_client import OpenRouterClient
from icn.training.trainer import ICNTrainer


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    Path("logs").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/icn_benchmark.log'),
            logging.StreamHandler()
        ]
    )


async def main():
    """Main benchmark execution."""
    
    parser = argparse.ArgumentParser(description="ICN Malicious Package Detection Benchmark")
    
    # Data preparation
    parser.add_argument("--prepare-data", action="store_true",
                       help="Prepare benchmark data from ICN pipeline")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per category (for testing)")
    parser.add_argument("--use-existing-eval", action="store_true",
                       help="Use existing eval split from training")
    
    # Model selection
    parser.add_argument("--icn-checkpoint", type=str, 
                       help="Path to ICN model checkpoint")
    parser.add_argument("--include-huggingface", action="store_true",
                       help="Include HuggingFace models (Endor Labs BERT)")
    parser.add_argument("--include-llms", action="store_true",
                       help="Include LLM models via OpenRouter")
    parser.add_argument("--include-baselines", action="store_true",
                       help="Include traditional baseline methods")
    
    # LLM configuration
    parser.add_argument("--openrouter-api-key", type=str,
                       help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--llm-models", nargs="+", 
                       default=["openai/gpt-4o", "anthropic/claude-3.5-sonnet"],
                       help="OpenRouter model IDs to test")
    
    # Execution parameters
    parser.add_argument("--max-concurrent", type=int, default=3,
                       help="Maximum concurrent model evaluations")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    # Quick test options
    parser.add_argument("--quick-test", action="store_true",
                       help="Run quick test with limited samples and models")
    parser.add_argument("--dry-run", action="store_true",
                       help="Prepare data and setup models without running benchmark")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting ICN Comprehensive Benchmark Study")
    logger.info(f"   Max samples per category: {args.max_samples}")
    logger.info(f"   Output directory: {args.output_dir}")
    logger.info(f"   Quick test mode: {args.quick_test}")
    
    try:
        # Phase 1: Data Preparation
        logger.info("📊 Phase 1: Benchmark Data Preparation")
        
        data_preparator = BenchmarkDataPreparator()
        
        if args.prepare_data or not Path("data/benchmark_samples/test_samples.json").exists():
            
            if args.use_existing_eval:
                # Use existing training split
                logger.info("📤 Using existing eval split from ICN training...")
                # This would require loading the pre-split data
                # For now, we'll create fresh data
                train_samples, test_samples = data_preparator.create_benchmark_split_from_icn_data(
                    max_samples_per_category=args.max_samples or (100 if args.quick_test else None),
                    test_split_ratio=0.2,
                    force_recompute=True
                )
            else:
                # Create fresh benchmark data
                train_samples, test_samples = data_preparator.create_benchmark_split_from_icn_data(
                    max_samples_per_category=args.max_samples or (100 if args.quick_test else None),
                    test_split_ratio=0.2
                )
            
            # Save benchmark samples
            output_dir = Path("data/benchmark_samples")
            output_dir.mkdir(parents=True, exist_ok=True)
            data_preparator.save_benchmark_samples(test_samples, output_dir / "test_samples.json")
            
        else:
            # Load existing benchmark data
            logger.info("📥 Loading existing benchmark samples...")
            test_samples = data_preparator.load_benchmark_samples(Path("data/benchmark_samples/test_samples.json"))
        
        if not test_samples:
            logger.error("❌ No benchmark samples available!")
            return
        
        # Phase 2: Model Registration
        logger.info("🤖 Phase 2: Model Registration")
        
        benchmark = BenchmarkSuite(output_dir=args.output_dir)
        benchmark.load_samples(test_samples)
        
        models_registered = 0
        
        # Register ICN model
        if args.icn_checkpoint and Path(args.icn_checkpoint).exists():
            logger.info(f"📝 Registering ICN model: {args.icn_checkpoint}")
            icn_model = ICNBenchmarkModel(args.icn_checkpoint)
            benchmark.register_model(icn_model)
            models_registered += 1
        else:
            logger.warning("⚠️  ICN checkpoint not provided or not found")
        
        # Register HuggingFace models
        if args.include_huggingface:
            logger.info("📝 Registering HuggingFace models...")
            try:
                endor_model = HuggingFaceModel(
                    model_name="Endor_Labs_BERT",
                    model_id="endorlabs/malicious-package-classifier-bert-mal-only"
                )
                benchmark.register_model(endor_model)
                models_registered += 1
            except Exception as e:
                logger.warning(f"⚠️  Failed to load Endor Labs BERT: {e}")
        
        # Register OpenRouter LLM models
        if args.include_llms:
            # Check API key
            openrouter_key = args.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
            if not openrouter_key:
                logger.warning("⚠️  OpenRouter API key not provided, skipping LLM models")
            else:
                logger.info("📝 Registering OpenRouter LLM models...")
                
                # Limit models for quick test
                llm_models = args.llm_models
                if args.quick_test:
                    llm_models = llm_models[:2]  # Only test first 2 models
                
                for model_id in llm_models:
                    model_name = f"LLM_{model_id.replace('/', '_')}"
                    llm_model = OpenRouterModel(
                        model_name=model_name,
                        openrouter_model_id=model_id,
                        prompt_type="zero_shot"
                    )
                    benchmark.register_model(llm_model)
                    models_registered += 1
        
        # Register baseline models
        if args.include_baselines:
            logger.info("📝 Registering baseline models...")
            
            heuristic_model = BaselineModel("Heuristic_Baseline", "heuristic")
            benchmark.register_model(heuristic_model)
            models_registered += 1
            
            random_model = BaselineModel("Random_Baseline", "random")
            benchmark.register_model(random_model)
            models_registered += 1
        
        if models_registered == 0:
            logger.error("❌ No models registered for benchmarking!")
            logger.error("   Use --icn-checkpoint, --include-huggingface, --include-llms, or --include-baselines")
            return
        
        logger.info(f"✅ Registered {models_registered} models for benchmarking")
        
        if args.dry_run:
            logger.info("🔍 Dry run complete - exiting before benchmark execution")
            return
        
        # Phase 3: Benchmark Execution
        logger.info("🚀 Phase 3: Benchmark Execution")
        
        # Setup OpenRouter client if needed
        openrouter_client = None
        if args.include_llms and (args.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")):
            openrouter_client = OpenRouterClient()
        
        # Run benchmark
        if openrouter_client:
            async with openrouter_client:
                results_df = await benchmark.run_benchmark(max_concurrent=args.max_concurrent)
        else:
            results_df = await benchmark.run_benchmark(max_concurrent=args.max_concurrent)
        
        # Phase 4: Analysis and Reporting
        logger.info("📊 Phase 4: Results Analysis and Reporting")
        
        # Compute metrics
        metrics = benchmark.compute_metrics()
        
        # Generate report
        report = benchmark.generate_report()
        
        # Save results
        benchmark.save_results("benchmark_results_detailed.json")
        
        # Save metrics summary
        metrics_file = Path(args.output_dir) / "metrics_summary.json"
        import json
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Save report
        report_file = Path(args.output_dir) / "benchmark_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Print summary to console
        print("\n" + "="*80)
        print("🎯 ICN BENCHMARK STUDY COMPLETE")
        print("="*80)
        print(report)
        print("="*80)
        print(f"📁 Detailed results saved to: {args.output_dir}")
        print(f"📊 Metrics summary: {metrics_file}")
        print(f"📝 Full report: {report_file}")
        
        # Print top performers
        if metrics:
            print(f"\n🏆 TOP PERFORMERS:")
            f1_scores = {}
            for model_name, model_metrics in metrics.items():
                f1_key = f"{model_name}_f1"
                if f1_key in model_metrics:
                    f1_scores[model_name] = model_metrics[f1_key]
            
            # Sort by F1 score
            sorted_models = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
            for i, (model_name, f1_score) in enumerate(sorted_models[:3], 1):
                print(f"   {i}. {model_name}: F1 = {f1_score:.3f}")
        
        print("="*80)
        logger.info("🎉 Benchmark study completed successfully!")
        
        # Print OpenRouter usage if applicable
        if openrouter_client:
            usage_summary = openrouter_client.get_usage_summary()
            logger.info(f"💰 OpenRouter API usage:")
            logger.info(f"   Total cost: ${usage_summary['total_cost_usd']:.2f}")
            logger.info(f"   Total requests: {usage_summary['total_requests']}")
            logger.info(f"   Average latency: {usage_summary['average_latency_seconds']:.2f}s")
        
    except KeyboardInterrupt:
        logger.info("⚠️  Benchmark interrupted by user")
        
    except Exception as e:
        logger.error(f"💥 Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Run the benchmark
    asyncio.run(main())