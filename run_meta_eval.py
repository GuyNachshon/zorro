#!/usr/bin/env python3
"""
Quick runner for Zorro Framework Meta-Evaluation
Simple interface to compare all models.
"""

import asyncio
import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from meta_evaluation import MetaEvaluator, MetaEvaluationConfig


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


async def run_quick_evaluation():
    """Run a quick evaluation with sample data."""
    
    print("🚀 Running Zorro Framework Meta-Evaluation (Demo Mode)")
    print("=" * 60)
    
    config = MetaEvaluationConfig(
        models_to_evaluate=["ICN", "AMIL", "CPG-GNN", "NeoBERT"],
        max_samples_per_model=50,  # Small for demo
        include_llm_comparison=False,
        include_baseline_comparison=True,
        output_dir="evaluation_results",
        generate_plots=True,
        compute_significance=True
    )
    
    evaluator = MetaEvaluator(config)
    
    try:
        results = await evaluator.run_complete_evaluation()
        
        print("\n🎉 Meta-Evaluation Completed!")
        print("=" * 40)
        
        # Print quick summary
        print("\n📊 PERFORMANCE RANKING:")
        for i, model in enumerate(results.performance_ranking):
            model_result = results.model_results[model]
            print(f"  {i+1}. {model:<12} F1: {model_result.f1_score:.3f} "
                 f"AUC: {model_result.roc_auc:.3f}")
        
        print("\n⚡ SPEED RANKING:")
        for i, model in enumerate(results.speed_ranking):
            model_result = results.model_results[model]
            print(f"  {i+1}. {model:<12} Time: {model_result.avg_inference_time:.3f}s")
        
        print(f"\n🏆 WINNERS:")
        print(f"  Best Overall:  {results.best_overall_model}")
        print(f"  Fastest:       {results.best_speed_model}")
        
        print(f"\n📁 Detailed results saved to: {config.output_dir}/")
        print(f"   📋 Summary:     evaluation_summary.txt")
        print(f"   📊 Detailed:    detailed_evaluation_report.md") 
        print(f"   📈 Plots:       model_comparison.png, speed_vs_performance.png")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def run_custom_evaluation(args):
    """Run evaluation with custom parameters."""
    
    print("🚀 Running Custom Zorro Framework Meta-Evaluation")
    print("=" * 60)
    
    # Check for model checkpoints
    model_paths = {}
    if args.icn_model:
        model_paths['icn_model_path'] = args.icn_model
    if args.amil_model:
        model_paths['amil_model_path'] = args.amil_model
    if args.cpg_model:
        model_paths['cpg_model_path'] = args.cpg_model
    if args.neobert_model:
        model_paths['neobert_model_path'] = args.neobert_model
    
    config = MetaEvaluationConfig(
        models_to_evaluate=args.models,
        test_data_path=args.test_data,
        max_samples_per_model=args.max_samples,
        include_llm_comparison=args.include_llms,
        include_baseline_comparison=not args.no_baselines,
        output_dir=args.output_dir,
        generate_plots=not args.no_plots,
        compute_significance=not args.no_stats,
        **model_paths
    )
    
    evaluator = MetaEvaluator(config)
    
    try:
        results = await evaluator.run_complete_evaluation()
        
        print(f"\n🎉 Custom evaluation completed!")
        print(f"📁 Results: {config.output_dir}/")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Custom evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(
        description="Zorro Framework Meta-Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick demo evaluation
  python run_meta_eval.py --quick
  
  # Custom evaluation with specific models
  python run_meta_eval.py --models ICN AMIL --max-samples 200
  
  # Full evaluation with trained models
  python run_meta_eval.py --icn-model checkpoints/icn/best.pth \\
                          --amil-model checkpoints/amil/best.pth \\
                          --test-data data/test_samples.json
        """
    )
    
    # Mode selection
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick demo evaluation")
    
    # Model selection
    parser.add_argument("--models", nargs="+", 
                       choices=["ICN", "AMIL", "CPG-GNN", "NeoBERT"],
                       default=["ICN", "AMIL", "CPG-GNN", "NeoBERT"],
                       help="Models to evaluate")
    
    # Model checkpoints
    parser.add_argument("--icn-model", type=str, help="ICN model checkpoint path")
    parser.add_argument("--amil-model", type=str, help="AMIL model checkpoint path") 
    parser.add_argument("--cpg-model", type=str, help="CPG-GNN model checkpoint path")
    parser.add_argument("--neobert-model", type=str, help="NeoBERT model checkpoint path")
    
    # Data and evaluation settings
    parser.add_argument("--test-data", type=str, 
                       help="Test data file path")
    parser.add_argument("--max-samples", type=int, default=100,
                       help="Maximum samples per model")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    
    # Comparison options
    parser.add_argument("--include-llms", action="store_true",
                       help="Include LLM model comparisons (requires OpenRouter API)")
    parser.add_argument("--no-baselines", action="store_true",
                       help="Skip baseline model comparisons")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip plot generation")
    parser.add_argument("--no-stats", action="store_true",
                       help="Skip statistical significance testing")
    
    # Utility options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Run evaluation
    if args.quick:
        print("Running quick demo evaluation...")
        results = asyncio.run(run_quick_evaluation())
    else:
        print("Running custom evaluation...")
        results = asyncio.run(run_custom_evaluation(args))
    
    if results:
        print("\n✅ Evaluation completed successfully!")
        print(f"🏆 Best model: {results.best_overall_model}")
        if hasattr(results.model_results[results.best_overall_model], 'f1_score'):
            best_f1 = results.model_results[results.best_overall_model].f1_score
            print(f"📊 Best F1 Score: {best_f1:.3f}")
    else:
        print("\n❌ Evaluation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()