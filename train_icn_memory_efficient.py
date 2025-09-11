#!/usr/bin/env python3
"""
Memory-efficient ICN training with proper GPU memory management.
"""

import sys
from pathlib import Path
import torch
import logging
import gc
import os

# Set memory management environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from train_icn_cached import load_cached_dataset, fix_package_tensors
from icn.training.trainer import ICNTrainer
from icn.training.config import create_training_config
from icn.models.icn_model import ICNModel

def clear_memory():
    """Clear GPU memory between stages."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def main():
    """Run memory-efficient ICN training."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("🚀 Memory-Efficient ICN Training")
    print("=" * 50)
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["tiny", "small", "medium", "full"], 
                       default="small", help="Dataset size to use")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (smaller = less memory)")
    parser.add_argument("--max-units", type=int, default=20, help="Max units per package (smaller = less memory)")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps")
    args = parser.parse_args()
    
    # Load cached dataset
    try:
        train_packages = load_cached_dataset(args.dataset)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return False
    
    # Split into train/eval
    n_eval = min(50, len(train_packages) // 10)  # Smaller eval set
    eval_packages = train_packages[:n_eval]
    train_packages = train_packages[n_eval:]
    
    print(f"\n📊 Dataset split:")
    print(f"  Train: {len(train_packages)} packages")
    print(f"  Eval: {len(eval_packages)} packages")
    
    # Create memory-efficient config
    config = create_training_config(
        experiment_name=f"icn_{args.dataset}_memory_efficient",
        batch_size=args.batch_size,  # Smaller batch size
        learning_rate=2e-5,
        max_epochs=4,  # Fewer epochs per stage
        use_gpu=torch.cuda.is_available()
    )
    
    # Override for memory efficiency
    config.dataloader_num_workers = 0
    config.report_to = []
    config.logging_steps = 50
    config.eval_steps = 100
    config.save_steps = 200
    config.gradient_accumulation_steps = args.gradient_accumulation
    config.max_units_per_package = args.max_units  # Limit units
    config.use_mixed_precision = True  # Use mixed precision for memory efficiency
    
    # Modify curriculum to be lighter
    config.curriculum_stages = [
        {"name": "stage_a_pretraining", "epochs": 2, "lr_factor": 1.0},
        {"name": "stage_b_convergence", "epochs": 2, "lr_factor": 0.5},
        {"name": "stage_c_malicious", "epochs": 3, "lr_factor": 0.25},
        {"name": "stage_d_robustness", "epochs": 2, "lr_factor": 0.1},
    ]
    
    # Create model with memory efficiency in mind
    print("\n🏗️  Creating ICN model...")
    model = ICNModel(
        vocab_size=50265,
        embedding_dim=config.embedding_dim,
        n_fixed_intents=config.n_fixed_intents,
        n_latent_intents=config.n_latent_intents,
        hidden_dim=256,  # Smaller hidden dim
        max_seq_length=256,  # Shorter sequences
        max_iterations=4,  # Fewer convergence iterations
        convergence_threshold=0.01,
        use_pretrained=False,  # Don't use pretrained for memory
        model_name="microsoft/codebert-base"
    )
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✅ Model created and moved to {device}")
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model.local_estimator, 'encoder'):
        if hasattr(model.local_estimator.encoder, 'gradient_checkpointing_enable'):
            model.local_estimator.encoder.gradient_checkpointing_enable()
            print("✅ Gradient checkpointing enabled")
    
    # Clear memory before training
    clear_memory()
    print(f"🧹 GPU memory cleared")
    
    # Create trainer with memory management hooks
    print("\n🏋️  Creating trainer...")
    trainer = ICNTrainer(
        model=model,
        config=config,
        train_packages=train_packages,
        eval_packages=eval_packages
    )
    
    # Override trainer to add memory clearing between stages
    original_train_stage = trainer.train_stage
    
    def train_stage_with_memory_clear(stage_name):
        result = original_train_stage(stage_name)
        clear_memory()  # Clear memory after each stage
        print(f"🧹 Memory cleared after {stage_name}")
        return result
    
    trainer.train_stage = train_stage_with_memory_clear
    
    # Train with memory monitoring
    print("\n🚂 Starting memory-efficient training...")
    
    try:
        trainer.train_curriculum()
        print("\n✅ Training complete!")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ GPU OOM Error: {e}")
        print("\nTips to reduce memory usage:")
        print("  1. Reduce batch size: --batch-size 1")
        print("  2. Reduce max units: --max-units 10")
        print("  3. Increase gradient accumulation: --gradient-accumulation 8")
        print("  4. Use smaller dataset: --dataset tiny")
        return False
    
    finally:
        # Clean up
        clear_memory()
        if 'model' in locals():
            del model
        if 'trainer' in locals():
            del trainer
        clear_memory()
        print("🧹 Final cleanup completed")
    
    return True

if __name__ == "__main__":
    # Clear memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    success = main()
    sys.exit(0 if success else 1)