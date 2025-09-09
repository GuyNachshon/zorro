"""
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
