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
from icn.training.config import create_training_config

def main():
    """Run quick ICN training test with limited data."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("🚀 Starting Quick ICN Training Test")
    print("=" * 50)
    
    # Create config with limited data for testing
    config = create_training_config(
        experiment_name="icn_quick_test",
        batch_size=2,  # Very small batch for quick test
        learning_rate=2e-5,
        max_epochs=1,  # Just 1 epoch for testing
        use_gpu=True
    )
    
    # Override some settings for quick test
    config.dataloader_num_workers = 0  # Avoid multiprocessing issues
    config.report_to = []  # Disable wandb
    config.logging_steps = 1
    config.eval_steps = 10
    config.save_steps = 50
    
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