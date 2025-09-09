#!/usr/bin/env python3
"""
Quick ICN training test with limited data to verify fixes.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from icn.training.trainer import ICNTrainer
from icn.training.config import ICNConfig

def main():
    """Run quick ICN training test with limited data."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("🚀 Starting Quick ICN Training Test")
    print("=" * 50)
    
    # Create config with limited data for testing
    config = ICNConfig(
        # Model config
        vocab_size=50265,
        embedding_dim=768,
        n_fixed_intents=15,
        n_latent_intents=10,
        hidden_dim=512,
        max_seq_length=512,
        max_iterations=6,
        convergence_threshold=0.01,
        use_pretrained=True,
        model_name="microsoft/codebert-base",
        
        # Training config - MUCH smaller for quick test
        batch_size=2,
        learning_rate=2e-5,
        encoder_lr=1e-5,
        weight_decay=0.01,
        num_epochs=1,  # Just 1 epoch for testing
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        
        # Data config - VERY limited data for quick test
        benign_samples=50,     # Much smaller
        malicious_samples=50,  # Much smaller
        test_split=0.2,
        val_split=0.1,
        max_units_per_package=20,
        
        # Quick curriculum - skip most stages for testing
        curriculum_stages=[
            {"name": "intent_pretraining", "epochs": 1, "sample_types": ["benign"]},
            {"name": "malicious_training", "epochs": 1, "sample_types": ["benign", "malicious_intent"]},
        ],
        
        # Hardware/logging
        use_cuda=True,
        use_mixed_precision=True,
        checkpoint_dir="./checkpoints",
        log_dir="./logs",
        use_wandb=False,  # Disable wandb for quick test
        save_every_epochs=1,
        eval_every_epochs=1
    )
    
    try:
        trainer = ICNTrainer(config)
        print("✅ ICN Trainer created successfully")
        
        # Test data loading
        print("\n📂 Testing data loading...")
        trainer.prepare_data()
        print("✅ Data loading completed")
        
        # Test model setup
        print("\n🏗️  Testing model setup...")
        trainer.setup_model()
        print("✅ Model setup completed")
        
        # Test one training step
        print("\n🏋️  Testing training step...")
        trainer.train()
        print("✅ Training completed successfully!")
        
        print("\n🎉 All tests passed! ICN training pipeline is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)