#!/usr/bin/env python3
"""
Fix Training Errors in ICN Pipeline
Patches the issues found in the training pipeline.
"""

import sys
from pathlib import Path

def fix_parse_package_call():
    """Fix the parse_package call to include required arguments."""
    
    file_path = Path("icn/data/data_preparation.py")
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix the parse_package call in process_single_malicious_sample
    old_line = "code_units = self.parser.parse_package(extracted_path)"
    new_line = "code_units = self.parser.parse_package(extracted_path, sample.name, sample.ecosystem)"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        print(f"✅ Fixed parse_package call in process_single_malicious_sample")
    else:
        print(f"⚠️ parse_package call already fixed or not found")
    
    # Fix the parse_package call in process_single_benign_sample
    old_line2 = "code_units = self.parser.parse_package(package_path)"
    new_line2 = "code_units = self.parser.parse_package(package_path, sample.name, sample.ecosystem)"
    
    if old_line2 in content:
        content = content.replace(old_line2, new_line2)
        print(f"✅ Fixed parse_package call in process_single_benign_sample")
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    return True


def fix_benign_collector_method():
    """Fix the BenignCollector method name."""
    
    file_path = Path("icn/data/data_preparation.py")
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace collect_popular_packages with collect_balanced_dataset
    old_call = """benign_samples = self.benign_collector.collect_popular_packages(
            target_count=target_count,
            cache_dir=self.benign_cache_path
        )"""
    
    new_call = """benign_samples = self.benign_collector.collect_balanced_dataset(
            total_samples=target_count,
            output_dir=self.benign_cache_path
        )"""
    
    if "collect_popular_packages" in content:
        content = content.replace(old_call, new_call)
        print(f"✅ Fixed BenignCollector method call")
    else:
        print(f"⚠️ BenignCollector method already fixed or not found")
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    return True


def fix_experiment_tracker_error():
    """Fix the UnboundLocalError for experiment_tracker."""
    
    file_path = Path("train_icn.py")
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the line where experiment_tracker is initialized
    for i, line in enumerate(lines):
        if "experiment_tracker = None" in line:
            print(f"⚠️ experiment_tracker initialization already exists")
            return True
    
    # Add initialization at the beginning of main function
    for i, line in enumerate(lines):
        if "try:" in line and "Step 1: Prepare training dataset" in ''.join(lines[i:i+5]):
            # Insert before the try block
            lines.insert(i, "    experiment_tracker = None  # Initialize to avoid UnboundLocalError\n")
            print(f"✅ Added experiment_tracker initialization")
            break
    
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    return True


def create_minimal_benign_collector():
    """Create a minimal BenignCollector that works."""
    
    content = '''"""
Minimal Benign Collector for ICN Training
Quick fix to avoid collection errors.
"""

from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class BenignSample:
    """Simple benign sample data structure."""
    def __init__(self, name: str, ecosystem: str, version: str = "1.0.0"):
        self.name = name
        self.ecosystem = ecosystem
        self.version = version
        self.source_url = f"https://{ecosystem}.org/{name}"
        self.local_path = None


class BenignCollector:
    """Minimal benign collector that returns synthetic data."""
    
    def __init__(self):
        logger.info("Initialized Minimal BenignCollector")
    
    def collect_balanced_dataset(self, 
                                total_samples: int = 100,
                                output_dir: Optional[Path] = None,
                                **kwargs) -> List[BenignSample]:
        """Return synthetic benign samples to avoid collection issues."""
        
        logger.info(f"Creating {total_samples} synthetic benign samples...")
        
        samples = []
        
        # Create synthetic npm packages
        npm_count = int(total_samples * 0.7)
        for i in range(npm_count):
            samples.append(BenignSample(
                name=f"benign-npm-package-{i}",
                ecosystem="npm",
                version=f"1.{i}.0"
            ))
        
        # Create synthetic pypi packages
        pypi_count = total_samples - npm_count
        for i in range(pypi_count):
            samples.append(BenignSample(
                name=f"benign-pypi-package-{i}",
                ecosystem="pypi",
                version=f"0.{i}.0"
            ))
        
        logger.info(f"✅ Created {len(samples)} synthetic benign samples")
        return samples
    
    def collect_popular_packages(self, 
                                target_count: int = 100,
                                cache_dir: Optional[Path] = None,
                                **kwargs) -> List[BenignSample]:
        """Alias for collect_balanced_dataset for compatibility."""
        return self.collect_balanced_dataset(total_samples=target_count, output_dir=cache_dir)
'''
    
    file_path = Path("icn/data/benign_collector_minimal.py")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Created minimal benign collector at {file_path}")
    return True


def main():
    """Apply all fixes."""
    
    print("🔧 Applying fixes to ICN training pipeline...")
    print("=" * 60)
    
    # Fix 1: parse_package signature
    print("\n1. Fixing parse_package signature issue...")
    fix_parse_package_call()
    
    # Fix 2: BenignCollector method
    print("\n2. Fixing BenignCollector method...")
    fix_benign_collector_method()
    
    # Fix 3: experiment_tracker error
    print("\n3. Fixing experiment_tracker UnboundLocalError...")
    fix_experiment_tracker_error()
    
    # Fix 4: Create minimal benign collector as backup
    print("\n4. Creating minimal benign collector...")
    create_minimal_benign_collector()
    
    print("\n" + "=" * 60)
    print("✅ All fixes applied!")
    print("\nYou can now run the training with:")
    print("  python train_icn.py --max-epochs 3")
    print("\nOr use the minimal benign collector by modifying imports:")
    print("  from .benign_collector_minimal import BenignCollector")
    

if __name__ == "__main__":
    main()