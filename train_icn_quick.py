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
from icn.models.icn_model import ICNModel
from icn.data.data_preparation import ICNDataPreparator

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
        # Create model
        print("\n🏗️  Creating ICN model...")
        model = ICNModel(
            vocab_size=50265,
            embedding_dim=config.embedding_dim,
            n_fixed_intents=config.n_fixed_intents,
            n_latent_intents=config.n_latent_intents,
            hidden_dim=config.hidden_dim,
            max_seq_length=config.max_seq_length,
            max_iterations=config.max_convergence_iterations,
            convergence_threshold=config.convergence_threshold,
            use_pretrained=config.use_pretrained,
            model_name=config.model_name
        )
        print("✅ ICN model created successfully")
        
        # Prepare minimal data
        print("\n📂 Preparing minimal test data...")
        data_preparator = ICNDataPreparator(
            malicious_dataset_path="malicious-software-packages-dataset",
            benign_cache_path="data/benign_samples"
        )
        
        # Get just a few samples for quick testing
        print("  Loading and processing 5 benign samples...")
        raw_benign = data_preparator.get_benign_samples(target_count=5)
        benign_processed = []
        for sample in raw_benign[:5]:  # Take first 5
            processed = data_preparator.process_single_benign_sample(sample)
            if processed:
                benign_processed.append(processed)
        
        print("  Loading and processing 5 malicious samples...")
        raw_malicious_dict = data_preparator.get_malicious_samples(max_samples=5)
        malicious_processed = []
        for samples_list in raw_malicious_dict.values():
            for sample in samples_list:
                if len(malicious_processed) >= 5:
                    break
                processed = data_preparator.process_single_malicious_sample(sample)
                if processed:
                    malicious_processed.append(processed)
            if len(malicious_processed) >= 5:
                break
        
        # Combine for training
        train_packages = benign_processed + malicious_processed
        print(f"✅ Prepared {len(train_packages)} total processed packages")
        print(f"   Benign: {len(benign_processed)}, Malicious: {len(malicious_processed)}")
        
        # Create trainer
        print("\n🏋️  Creating ICN trainer...")
        trainer = ICNTrainer(
            model=model,
            config=config,
            train_packages=train_packages,
            eval_packages=train_packages[:5]  # Use first 5 for eval
        )
        print("✅ ICN Trainer created successfully")
        
        # Test one training epoch
        print("\n🏋️  Testing training step...")
        trainer.train_curriculum()
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