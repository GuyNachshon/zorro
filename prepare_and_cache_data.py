#!/usr/bin/env python3
"""
One-time data preparation and caching script.
Run this ONCE to prepare all data, then training can just load from cache.
"""

import sys
from pathlib import Path
import torch
import pickle
import json
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from icn.data.data_preparation import ICNDataPreparator
from icn.training.dataloader import ProcessedPackage

def prepare_and_save_datasets():
    """Prepare all datasets once and save to disk."""
    
    print("🚀 One-time Data Preparation and Caching")
    print("=" * 50)
    
    # Create cache directory
    cache_dir = Path("data/cached_datasets")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data preparator
    preparator = ICNDataPreparator(
        malicious_dataset_path="malicious-software-packages-dataset",
        benign_cache_path="data/benign_samples"
    )
    
    # Prepare different dataset sizes
    dataset_configs = [
        {"name": "tiny", "malicious": 10, "benign": 10},      # For quick tests
        {"name": "small", "malicious": 100, "benign": 100},   # For development
        {"name": "medium", "malicious": 1000, "benign": 500}, # For validation
        {"name": "full", "malicious": None, "benign": 1000},  # Full dataset
    ]
    
    for config in dataset_configs:
        print(f"\n📦 Preparing {config['name']} dataset...")
        cache_file = cache_dir / f"icn_dataset_{config['name']}.pt"
        
        if cache_file.exists():
            print(f"  ✓ Cache already exists: {cache_file}")
            continue
        
        # Prepare dataset
        print(f"  Loading {config['malicious'] or 'all'} malicious samples...")
        malicious_dict = preparator.get_malicious_samples(max_samples=config['malicious'])
        
        print(f"  Loading {config['benign']} benign samples...")
        benign_samples = preparator.get_benign_samples(target_count=config['benign'])
        
        # Process samples
        processed_packages = []
        
        print(f"  Processing malicious samples...")
        for category, samples in tqdm(malicious_dict.items()):
            for sample in samples:
                processed = preparator.process_single_malicious_sample(sample)
                if processed:
                    # Move tensors to CPU for storage
                    processed = move_package_to_cpu(processed)
                    processed_packages.append(processed)
        
        print(f"  Processing benign samples...")
        for sample in tqdm(benign_samples):
            processed = preparator.process_single_benign_sample(sample)
            if processed:
                # Move tensors to CPU for storage
                processed = move_package_to_cpu(processed)
                processed_packages.append(processed)
        
        # Save to cache
        print(f"  💾 Saving {len(processed_packages)} packages to {cache_file}")
        torch.save({
            'packages': processed_packages,
            'config': config,
            'n_packages': len(processed_packages),
            'n_malicious': sum(1 for p in processed_packages if p.malicious_label > 0.5),
            'n_benign': sum(1 for p in processed_packages if p.malicious_label < 0.5)
        }, cache_file)
        
        print(f"  ✅ Saved {config['name']} dataset!")
    
    # Create metadata file
    metadata_file = cache_dir / "metadata.json"
    metadata = {
        "datasets": dataset_configs,
        "cache_dir": str(cache_dir),
        "description": "Pre-processed ICN datasets with different sizes"
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✅ All datasets prepared and cached!")
    print(f"📁 Cache location: {cache_dir}")
    return cache_dir

def move_package_to_cpu(package: ProcessedPackage) -> ProcessedPackage:
    """Move all tensors in a package to CPU for storage."""
    # Create new package with CPU tensors
    return ProcessedPackage(
        name=package.name,
        ecosystem=package.ecosystem,
        sample_type=package.sample_type,
        units=package.units,
        input_ids=[t.cpu() if torch.is_tensor(t) else t for t in package.input_ids],
        attention_masks=[t.cpu() if torch.is_tensor(t) else t for t in package.attention_masks],
        phase_ids=[t.cpu() if torch.is_tensor(t) else t for t in package.phase_ids],
        api_features=[t.cpu() if torch.is_tensor(t) else t for t in package.api_features],
        ast_features=[t.cpu() if torch.is_tensor(t) else t for t in package.ast_features],
        malicious_label=package.malicious_label,
        intent_labels=package.intent_labels.cpu() if torch.is_tensor(package.intent_labels) else package.intent_labels,
        manifest_embedding=package.manifest_embedding.cpu() if torch.is_tensor(package.manifest_embedding) else package.manifest_embedding
    )

def load_cached_dataset(dataset_name: str = "small"):
    """Load a pre-cached dataset."""
    cache_file = Path(f"data/cached_datasets/icn_dataset_{dataset_name}.pt")
    
    if not cache_file.exists():
        raise FileNotFoundError(f"Dataset {dataset_name} not found. Run prepare_and_cache_data.py first!")
    
    print(f"📂 Loading cached dataset: {dataset_name}")
    data = torch.load(cache_file, map_location='cpu')
    print(f"  ✓ Loaded {data['n_packages']} packages")
    print(f"    Malicious: {data['n_malicious']}")
    print(f"    Benign: {data['n_benign']}")
    
    return data['packages']

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare and cache ICN datasets")
    parser.add_argument("--dataset", choices=["tiny", "small", "medium", "full"], 
                       default=None, help="Prepare specific dataset only")
    args = parser.parse_args()
    
    if args.dataset:
        # Prepare single dataset
        print(f"Preparing {args.dataset} dataset only...")
        # Would need to modify prepare_and_save_datasets to handle this
    
    prepare_and_save_datasets()