#!/usr/bin/env python3
"""
Complete ICN Training Pipeline
Integrates malicious sample extraction with full curriculum training.
"""

import argparse
import logging
import os
from pathlib import Path
import torch

# ICN imports
from icn.models.icn_model import ICNModel
from icn.training.trainer import ICNTrainer
from icn.training.config import create_training_config, CurriculumConfig, validate_config
from icn.training.wandb_config import create_experiment_config, ExperimentTracker
from icn.evaluation.metrics import ICNMetrics


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/icn_training.log'),
            logging.StreamHandler()
        ]
    )


def create_icn_model(config) -> ICNModel:
    """Create and initialize ICN model."""
    
    model = ICNModel(
        model_name=config.model_name,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        n_fixed_intents=config.n_fixed_intents,
        n_latent_intents=config.n_latent_intents,
        max_seq_length=config.max_seq_length,
        max_convergence_iterations=config.max_convergence_iterations,
        convergence_threshold=config.convergence_threshold
    )
    
    # Move to appropriate device
    device = torch.device(config.device)
    model = model.to(device)
    
    logging.info(f"🤖 ICN Model created on {device}")
    logging.info(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logging.info(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model


def main():
    """Main training pipeline."""
    
    parser = argparse.ArgumentParser(description="ICN Malware Detection Training")
    parser.add_argument("--experiment-name", type=str, default="icn-malware-detection",
                       help="Experiment name for tracking")
    parser.add_argument("--malicious-dataset", type=str, 
                       default="malicious-software-packages-dataset",
                       help="Path to malicious dataset")
    parser.add_argument("--max-malicious-samples", type=int, default=None,
                       help="Maximum malicious samples (for testing)")
    parser.add_argument("--target-benign-count", type=int, default=5000,
                       help="Target number of benign samples")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--max-epochs", type=int, default=10,
                       help="Maximum epochs per stage")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--force-recompute", action="store_true",
                       help="Force recomputation of cached data")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable Weights & Biases tracking")
    parser.add_argument("--dry-run", action="store_true",
                       help="Prepare data and setup model without training")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting ICN Training Pipeline")
    logger.info(f"   Experiment: {args.experiment_name}")
    logger.info(f"   Malicious dataset: {args.malicious_dataset}")
    logger.info(f"   Max malicious samples: {args.max_malicious_samples}")
    logger.info(f"   Target benign samples: {args.target_benign_count}")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Learning rate: {args.learning_rate}")
    
    try:
        # Step 1: Prepare training dataset
        logger.info("📊 Phase 1: Dataset Preparation")
        train_packages, eval_packages = ICNTrainer.prepare_training_dataset(
            malicious_dataset_path=args.malicious_dataset,
            max_malicious_samples=args.max_malicious_samples,
            target_benign_count=args.target_benign_count,
            force_recompute=args.force_recompute,
            train_test_split=0.8
        )
        
        logger.info(f"✅ Dataset prepared: {len(train_packages)} train, {len(eval_packages)} eval")
        
        if args.dry_run:
            logger.info("🔍 Dry run complete - stopping before model training")
            return
        
        # Step 2: Create training configuration
        logger.info("⚙️  Phase 2: Configuration Setup")
        config = create_training_config(
            experiment_name=args.experiment_name,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            output_dir=f"checkpoints/{args.experiment_name}"
        )
        
        # Validate configuration
        config_issues = validate_config(config)
        if config_issues:
            logger.error("Configuration validation failed:")
            for issue in config_issues:
                logger.error(f"  - {issue}")
            return
        
        logger.info("✅ Configuration validated")
        
        # Step 3: Initialize experiment tracking
        experiment_tracker = None
        if not args.no_wandb:
            logger.info("📈 Phase 3: Experiment Tracking Setup")
            wandb_config = create_experiment_config(
                experiment_name=args.experiment_name,
                stage="training",
                tags=["icn", "curriculum-learning", "malware-detection"],
                notes=f"ICN training with {len(train_packages)} samples"
            )
            
            experiment_tracker = ExperimentTracker(
                config=wandb_config,
                training_config=config.__dict__
            )
        
        # Step 4: Create ICN model
        logger.info("🤖 Phase 4: Model Initialization")
        model = create_icn_model(config)
        
        # Step 5: Initialize trainer
        logger.info("🎓 Phase 5: Trainer Initialization")
        trainer = ICNTrainer(
            model=model,
            config=config,
            train_packages=train_packages,
            eval_packages=eval_packages,
            experiment_tracker=experiment_tracker
        )
        
        # Step 6: Start curriculum training
        logger.info("🚀 Phase 6: Curriculum Training")
        
        if experiment_tracker:
            experiment_tracker.init_experiment(model)
        
        # Train the full curriculum
        curriculum_results = trainer.train_curriculum()
        
        # Step 7: Final evaluation and summary
        logger.info("📊 Phase 7: Final Evaluation")
        
        # Perform comprehensive evaluation
        metrics_computer = ICNMetrics()
        
        # This would be expanded with actual evaluation logic
        logger.info("✅ Training pipeline completed successfully!")
        
        # Print summary
        print("\n" + "="*80)
        print("🎯 ICN TRAINING PIPELINE COMPLETE")
        print("="*80)
        print(f"📦 Dataset: {len(train_packages)} training + {len(eval_packages)} evaluation")
        print(f"🤖 Model: ICN with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"🎓 Stages: 4-stage curriculum learning")
        print(f"📊 Results saved to: {config.output_dir}")
        
        if experiment_tracker:
            print(f"📈 Experiment tracking: {experiment_tracker.run.url}")
            experiment_tracker.finish_experiment()
        
        print("="*80)
        
    except KeyboardInterrupt:
        logger.info("⚠️  Training interrupted by user")
        if experiment_tracker:
            experiment_tracker.finish_experiment()
            
    except Exception as e:
        logger.error(f"💥 Training pipeline failed: {e}")
        if experiment_tracker:
            experiment_tracker.finish_experiment()
        raise


if __name__ == "__main__":
    main()