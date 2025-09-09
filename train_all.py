#!/usr/bin/env python3
"""
Unified Training Runner for Zorro Framework
Train all models (ICN, AMIL, CPG-GNN, NeoBERT) with consistent interface.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent))

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/train_all.log'),
            logging.StreamHandler()
        ]
    )


async def run_training_script(script_path: str, args: list) -> Dict[str, Any]:
    """Run a training script with given arguments."""
    
    import subprocess
    
    cmd = [sys.executable, script_path] + args
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        result = {
            'script': script_path,
            'returncode': process.returncode,
            'stdout': stdout.decode('utf-8'),
            'stderr': stderr.decode('utf-8'),
            'success': process.returncode == 0
        }
        
        if result['success']:
            logger.info(f"✅ {script_path} completed successfully")
        else:
            logger.error(f"❌ {script_path} failed with code {process.returncode}")
            logger.error(f"STDERR: {result['stderr']}")
        
        return result
        
    except Exception as e:
        logger.error(f"💥 Failed to run {script_path}: {e}")
        return {
            'script': script_path,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'success': False
        }


async def train_all_models(args):
    """Train all models sequentially or in parallel."""
    
    logger.info("🚀 Starting unified training for all Zorro models...")
    logger.info("=" * 60)
    
    # Define training scripts and their arguments
    training_configs = []
    
    if "icn" in args.models:
        icn_args = [
            "--experiment-name", args.experiment_name or "zorro-icn",
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--max-epochs", str(args.max_epochs),
            "--log-level", args.log_level
        ]
        if args.data_path:
            icn_args.extend(["--malicious-dataset", args.data_path])
        if args.dry_run:
            icn_args.append("--dry-run")
        if args.no_wandb:
            icn_args.append("--no-wandb")
        
        training_configs.append(("train_icn.py", icn_args))
    
    if "amil" in args.models:
        amil_args = [
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--log-level", args.log_level,
            "--save-dir", f"checkpoints/{args.experiment_name or 'zorro'}-amil"
        ]
        if args.data_path:
            amil_args.extend(["--data-path", args.data_path])
        
        training_configs.append(("train_amil.py", amil_args))
    
    if "cpg" in args.models:
        cpg_args = [
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--log-level", args.log_level,
            "--save-dir", f"checkpoints/{args.experiment_name or 'zorro'}-cpg"
        ]
        if args.data_path:
            cpg_args.extend(["--data-path", args.data_path])
        
        training_configs.append(("train_cpg.py", cpg_args))
    
    if "neobert" in args.models:
        neobert_args = [
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
            "--max-length", str(args.max_length),
            "--pooling-strategy", args.pooling_strategy,
            "--log-level", args.log_level,
            "--save-dir", f"checkpoints/{args.experiment_name or 'zorro'}-neobert"
        ]
        if args.data_path:
            neobert_args.extend(["--data-path", args.data_path])
        
        training_configs.append(("train_neobert.py", neobert_args))
    
    if not training_configs:
        logger.error("No models selected for training!")
        return
    
    logger.info(f"Training {len(training_configs)} models: {', '.join(args.models)}")
    
    # Run training scripts
    if args.parallel:
        logger.info("Running training scripts in parallel...")
        tasks = [
            run_training_script(script, script_args) 
            for script, script_args in training_configs
        ]
        results = await asyncio.gather(*tasks)
    else:
        logger.info("Running training scripts sequentially...")
        results = []
        for script, script_args in training_configs:
            result = await run_training_script(script, script_args)
            results.append(result)
            
            # Stop on first failure if not in continue mode
            if not result['success'] and not args.continue_on_failure:
                logger.error("Stopping due to training failure")
                break
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("🎯 TRAINING SUMMARY")
    logger.info("=" * 60)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    logger.info(f"✅ Successful: {len(successful)}")
    logger.info(f"❌ Failed: {len(failed)}")
    
    for result in successful:
        logger.info(f"   ✓ {Path(result['script']).stem}")
    
    for result in failed:
        logger.error(f"   ✗ {Path(result['script']).stem}")
    
    if args.run_meta_eval and len(successful) > 1:
        logger.info("\n🔍 Running meta-evaluation...")
        meta_eval_args = [
            "--models", *[Path(r['script']).stem.replace('train_', '').upper() for r in successful],
            "--output-dir", f"evaluation_results/{args.experiment_name or 'zorro'}",
            "--max-samples", str(args.meta_eval_samples)
        ]
        
        meta_result = await run_training_script("run_meta_eval.py", meta_eval_args)
        
        if meta_result['success']:
            logger.info("✅ Meta-evaluation completed successfully!")
        else:
            logger.error("❌ Meta-evaluation failed")
    
    logger.info("=" * 60)
    
    return results


def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(
        description="Unified training runner for Zorro Framework models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models sequentially
  python train_all.py --models icn amil cpg neobert
  
  # Train specific models in parallel
  python train_all.py --models icn amil --parallel
  
  # Train with custom settings and run meta-evaluation
  python train_all.py --models icn amil cpg neobert \\
                      --batch-size 16 --learning-rate 1e-4 \\
                      --run-meta-eval --experiment-name "zorro-v1"
        """
    )
    
    # Model selection
    parser.add_argument("--models", nargs="+", 
                       choices=["icn", "amil", "cpg", "neobert"],
                       default=["icn", "amil", "cpg", "neobert"],
                       help="Models to train")
    
    # Training parameters
    parser.add_argument("--experiment-name", type=str, default="zorro",
                       help="Experiment name for organizing outputs")
    parser.add_argument("--data-path", type=str, 
                       help="Path to training data")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--max-epochs", type=int, default=10,
                       help="Maximum epochs per model")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Max sequence length (NeoBERT)")
    parser.add_argument("--pooling-strategy", type=str, default="attention",
                       choices=["mean", "attention", "mil"],
                       help="Pooling strategy (NeoBERT)")
    
    # Execution options
    parser.add_argument("--parallel", action="store_true",
                       help="Train models in parallel")
    parser.add_argument("--continue-on-failure", action="store_true",
                       help="Continue training other models if one fails")
    parser.add_argument("--dry-run", action="store_true",
                       help="Dry run mode (ICN only)")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable W&B tracking")
    
    # Meta-evaluation
    parser.add_argument("--run-meta-eval", action="store_true",
                       help="Run meta-evaluation after training")
    parser.add_argument("--meta-eval-samples", type=int, default=100,
                       help="Samples for meta-evaluation")
    
    # Utility
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Run training
    try:
        results = asyncio.run(train_all_models(args))
        
        successful_count = sum(1 for r in results if r['success'])
        
        if successful_count > 0:
            logger.info(f"🎉 Training completed! {successful_count}/{len(results)} models trained successfully")
        else:
            logger.error("💥 All training failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()