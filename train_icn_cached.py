#!/usr/bin/env python3
"""
ICN training using pre-cached datasets.
Much faster than recreating data every time!
"""

import sys
from pathlib import Path
import torch
import logging
from typing import List

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from icn.training.trainer import ICNTrainer
from icn.training.config import create_training_config
from icn.models.icn_model import ICNModel
from icn.training.dataloader import ProcessedPackage

def load_cached_dataset(dataset_name: str = "small") -> List[ProcessedPackage]:
    """Load a pre-cached dataset."""
    cache_file = Path(f"data/cached_datasets/icn_dataset_{dataset_name}.pt")
    
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found at {cache_file}.\n"
            f"Please run: python prepare_and_cache_data.py"
        )
    
    print(f"📂 Loading cached dataset: {dataset_name}")
    data = torch.load(cache_file, map_location='cpu', weights_only=False)
    print(f"  ✓ Loaded {data['n_packages']} packages")
    print(f"    Malicious: {data['n_malicious']}")
    print(f"    Benign: {data['n_benign']}")
    
    return data['packages']

def main():
    """Run ICN training with cached data."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("🚀 ICN Training with Cached Data")
    print("=" * 50)
    
    # Choose dataset size
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["tiny", "small", "medium", "full"], 
                       default="small", help="Dataset size to use")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()
    
    # Load cached dataset
    try:
        train_packages = load_cached_dataset(args.dataset)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return False
    
    # Split into train/eval
    n_eval = min(100, len(train_packages) // 10)
    eval_packages = train_packages[:n_eval]
    train_packages = train_packages[n_eval:]
    
    print(f"\n📊 Dataset split:")
    print(f"  Train: {len(train_packages)} packages")
    print(f"  Eval: {len(eval_packages)} packages")
    
    # Create config
    config = create_training_config(
        experiment_name=f"icn_{args.dataset}_cached",
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        use_gpu=torch.cuda.is_available()
    )
    
    # Override some settings for faster training
    config.dataloader_num_workers = 0  # Avoid multiprocessing issues
    config.report_to = []  # Disable wandb
    config.logging_steps = 10
    config.eval_steps = 50
    config.save_steps = 100
    
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
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✅ Model created and moved to {device}")
    
    # Create trainer
    print("\n🏋️  Creating trainer...")
    trainer = ICNTrainer(
        model=model,
        config=config,
        train_packages=train_packages,
        eval_packages=eval_packages
    )
    
    # Train!
    print("\n🚂 Starting training...")
    trainer.train_curriculum()
    
    print("\n✅ Training complete!")
    
    # Save final model
    final_checkpoint = Path(f"checkpoints/icn_{args.dataset}_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'dataset': args.dataset
    }, final_checkpoint)
    print(f"💾 Final model saved to {final_checkpoint}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)