"""
Prepare benchmark data from ICN processed packages.
Converts ProcessedPackage objects to BenchmarkSample format for model comparison.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import shutil

from ..training.dataloader import ProcessedPackage
from ..training.losses import SampleType
from ..data.data_preparation import ICNDataPreparator
from .benchmark_framework import BenchmarkSample

logger = logging.getLogger(__name__)


class BenchmarkDataPreparator:
    """Prepares ICN data for benchmarking against other models."""
    
    def __init__(self, 
                 malicious_dataset_path: str = "malicious-software-packages-dataset",
                 max_content_length: int = 50000):
        self.malicious_dataset_path = Path(malicious_dataset_path)
        self.max_content_length = max_content_length
        
        # Initialize ICN data preparator for raw access
        self.icn_preparator = ICNDataPreparator(malicious_dataset_path=str(malicious_dataset_path))
    
    def convert_processed_packages_to_benchmark(
        self, 
        processed_packages: List[ProcessedPackage]
    ) -> List[BenchmarkSample]:
        """
        Convert ICN ProcessedPackage objects to BenchmarkSample format.
        
        Args:
            processed_packages: List of ICN processed packages
            
        Returns:
            List of standardized benchmark samples
        """
        logger.info(f"🔄 Converting {len(processed_packages)} processed packages to benchmark format...")
        
        benchmark_samples = []
        successful_conversions = 0
        
        for i, pkg in enumerate(processed_packages):
            try:
                # Extract raw content for LLM analysis
                raw_content = self._extract_raw_content_from_package(pkg)
                individual_files = self._extract_individual_files_from_package(pkg)
                
                if not raw_content and not individual_files:
                    logger.warning(f"⚠️  No raw content found for {pkg.name}, skipping...")
                    continue
                
                # Map sample type to ground truth label
                ground_truth = 1 if pkg.sample_type != SampleType.BENIGN else 0
                
                # Create benchmark sample
                sample = BenchmarkSample(
                    package_name=pkg.name,
                    ecosystem=pkg.ecosystem,
                    sample_type=pkg.sample_type.value,
                    ground_truth_label=ground_truth,
                    raw_content=raw_content[:self.max_content_length] if raw_content else "",
                    file_paths=self._extract_file_paths(pkg),
                    individual_files=individual_files,
                    processed_package=pkg,
                    package_size_bytes=len(raw_content) if raw_content else 0,
                    num_files=len(pkg.units),
                    metadata={
                        "package_hash": pkg.package_hash,
                        "malicious_label": pkg.malicious_label
                    }
                )
                
                benchmark_samples.append(sample)
                successful_conversions += 1
                
                if (i + 1) % 100 == 0:
                    logger.info(f"   Converted {i + 1}/{len(processed_packages)} packages...")
                    
            except Exception as e:
                logger.error(f"❌ Failed to convert package {pkg.name}: {e}")
                continue
        
        logger.info(f"✅ Successfully converted {successful_conversions}/{len(processed_packages)} packages")
        
        # Print distribution
        self._print_benchmark_distribution(benchmark_samples)
        
        return benchmark_samples
    
    def _extract_raw_content_from_package(self, pkg: ProcessedPackage) -> str:
        """Extract raw text content from processed package."""
        try:
            # Combine raw content from all units
            content_parts = []
            
            for unit in pkg.units:
                if hasattr(unit, 'raw_content') and unit.raw_content:
                    # Add file header for context
                    file_info = f"\n# File: {unit.file_path} ({unit.unit_type})\n"
                    content_parts.append(file_info)
                    content_parts.append(unit.raw_content)
                    content_parts.append("\n" + "="*50 + "\n")
            
            combined_content = "".join(content_parts)
            return combined_content if combined_content.strip() else ""
            
        except Exception as e:
            logger.warning(f"Failed to extract raw content from {pkg.name}: {e}")
            return ""
    
    def _extract_individual_files_from_package(self, pkg: ProcessedPackage) -> Dict[str, str]:
        """Extract individual file contents from processed package."""
        individual_files = {}
        
        try:
            for unit in pkg.units:
                if hasattr(unit, 'raw_content') and unit.raw_content:
                    file_path = str(unit.file_path)
                    # Truncate individual files to reasonable size
                    content = unit.raw_content[:10000]  # 10KB limit per file
                    individual_files[file_path] = content
                    
        except Exception as e:
            logger.warning(f"Failed to extract individual files from {pkg.name}: {e}")
            
        return individual_files
    
    def _extract_file_paths(self, pkg: ProcessedPackage) -> List[str]:
        """Extract file paths from processed package."""
        try:
            return [str(unit.file_path) for unit in pkg.units if hasattr(unit, 'file_path')]
        except:
            return []
    
    def create_benchmark_split_from_icn_data(
        self,
        max_samples_per_category: Optional[int] = None,
        test_split_ratio: float = 0.2,
        force_recompute: bool = False
    ) -> Tuple[List[BenchmarkSample], List[BenchmarkSample]]:
        """
        Create train/test benchmark splits directly from ICN data pipeline.
        
        Args:
            max_samples_per_category: Limit samples per category for testing
            test_split_ratio: Fraction for test set
            force_recompute: Force recomputation of processed packages
            
        Returns:
            Tuple of (train_samples, test_samples)
        """
        logger.info("🔧 Creating benchmark splits from ICN data pipeline...")
        
        # Prepare processed packages using ICN pipeline
        target_malicious = max_samples_per_category if max_samples_per_category else None
        target_benign = max_samples_per_category * 2 if max_samples_per_category else 3000
        
        all_packages = self.icn_preparator.prepare_complete_dataset(
            max_malicious_samples=target_malicious,
            target_benign_count=target_benign,
            force_recompute=force_recompute
        )
        
        if not all_packages:
            raise ValueError("No packages were processed successfully!")
        
        # Convert to benchmark format
        all_samples = self.convert_processed_packages_to_benchmark(all_packages)
        
        # Split into train/test
        import random
        random.shuffle(all_samples)
        
        split_idx = int(len(all_samples) * (1 - test_split_ratio))
        train_samples = all_samples[:split_idx]
        test_samples = all_samples[split_idx:]
        
        logger.info(f"📊 Benchmark splits created:")
        logger.info(f"   Training samples: {len(train_samples)}")
        logger.info(f"   Test samples: {len(test_samples)}")
        
        return train_samples, test_samples
    
    def load_existing_eval_split(self, eval_packages: List[ProcessedPackage]) -> List[BenchmarkSample]:
        """
        Convert existing evaluation split to benchmark format.
        
        Args:
            eval_packages: Pre-split evaluation packages from ICN training
            
        Returns:
            List of benchmark samples for evaluation
        """
        logger.info(f"📤 Converting {len(eval_packages)} existing eval packages to benchmark format...")
        
        benchmark_samples = self.convert_processed_packages_to_benchmark(eval_packages)
        
        logger.info(f"✅ Converted {len(benchmark_samples)} eval packages for benchmarking")
        
        return benchmark_samples
    
    def _print_benchmark_distribution(self, samples: List[BenchmarkSample]):
        """Print distribution of benchmark samples."""
        distribution = {
            'total': len(samples),
            'by_ecosystem': {},
            'by_type': {},
            'by_label': {0: 0, 1: 0}
        }
        
        for sample in samples:
            # Count by ecosystem
            eco = sample.ecosystem
            distribution['by_ecosystem'][eco] = distribution['by_ecosystem'].get(eco, 0) + 1
            
            # Count by sample type
            stype = sample.sample_type
            distribution['by_type'][stype] = distribution['by_type'].get(stype, 0) + 1
            
            # Count by binary label
            distribution['by_label'][sample.ground_truth_label] += 1
        
        logger.info("📊 Benchmark sample distribution:")
        logger.info(f"   Total samples: {distribution['total']}")
        logger.info("   By ecosystem:")
        for eco, count in distribution['by_ecosystem'].items():
            logger.info(f"     {eco}: {count}")
        logger.info("   By sample type:")
        for stype, count in distribution['by_type'].items():
            logger.info(f"     {stype}: {count}")
        logger.info("   By label:")
        benign_count = distribution['by_label'][0]
        malicious_count = distribution['by_label'][1]
        total = benign_count + malicious_count
        logger.info(f"     Benign (0): {benign_count} ({benign_count/total*100:.1f}%)")
        logger.info(f"     Malicious (1): {malicious_count} ({malicious_count/total*100:.1f}%)")
    
    def save_benchmark_samples(
        self, 
        samples: List[BenchmarkSample], 
        filepath: Path,
        include_processed_packages: bool = False
    ):
        """
        Save benchmark samples to file.
        
        Args:
            samples: Benchmark samples to save
            filepath: Output file path
            include_processed_packages: Whether to include full ProcessedPackage objects
        """
        logger.info(f"💾 Saving {len(samples)} benchmark samples to {filepath}...")
        
        # Convert to serializable format
        serializable_samples = []
        for sample in samples:
            sample_data = {
                'package_name': sample.package_name,
                'ecosystem': sample.ecosystem,
                'sample_type': sample.sample_type,
                'ground_truth_label': sample.ground_truth_label,
                'raw_content': sample.raw_content,
                'file_paths': sample.file_paths,
                'individual_files': sample.individual_files,
                'package_size_bytes': sample.package_size_bytes,
                'num_files': sample.num_files,
                'metadata': sample.metadata
            }
            
            # Optionally include processed package data
            if include_processed_packages and sample.processed_package:
                sample_data['processed_package'] = {
                    'name': sample.processed_package.name,
                    'ecosystem': sample.processed_package.ecosystem,
                    'sample_type': sample.processed_package.sample_type.value,
                    'malicious_label': sample.processed_package.malicious_label,
                    'package_hash': sample.processed_package.package_hash,
                    'num_units': len(sample.processed_package.units)
                    # Note: Not including full tensors due to size
                }
            
            serializable_samples.append(sample_data)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': str(pd.Timestamp.now()),
                'sample_count': len(samples),
                'samples': serializable_samples
            }, f, indent=2)
        
        logger.info(f"✅ Benchmark samples saved to {filepath}")
    
    def load_benchmark_samples(self, filepath: Path) -> List[BenchmarkSample]:
        """
        Load benchmark samples from file.
        
        Args:
            filepath: Input file path
            
        Returns:
            List of loaded benchmark samples
        """
        logger.info(f"📥 Loading benchmark samples from {filepath}...")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        samples = []
        for sample_data in data['samples']:
            sample = BenchmarkSample(
                package_name=sample_data['package_name'],
                ecosystem=sample_data['ecosystem'],
                sample_type=sample_data['sample_type'],
                ground_truth_label=sample_data['ground_truth_label'],
                raw_content=sample_data['raw_content'],
                file_paths=sample_data['file_paths'],
                individual_files=sample_data.get('individual_files', {}),
                package_size_bytes=sample_data['package_size_bytes'],
                num_files=sample_data['num_files'],
                metadata=sample_data.get('metadata', {})
            )
            # Note: processed_package is not loaded from JSON
            samples.append(sample)
        
        logger.info(f"✅ Loaded {len(samples)} benchmark samples")
        
        return samples


# Import pandas for timestamp
import pandas as pd


if __name__ == "__main__":
    # Test benchmark data preparation
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    preparator = BenchmarkDataPreparator()
    
    # Create small test benchmark split
    try:
        train_samples, test_samples = preparator.create_benchmark_split_from_icn_data(
            max_samples_per_category=50,  # Small test set
            test_split_ratio=0.2
        )
        
        print(f"\n✅ Benchmark data preparation test complete:")
        print(f"   Training samples: {len(train_samples)}")
        print(f"   Test samples: {len(test_samples)}")
        
        # Save test samples
        output_dir = Path("data/benchmark_samples")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        preparator.save_benchmark_samples(test_samples, output_dir / "test_samples.json")
        
        print(f"   Saved test samples to: {output_dir / 'test_samples.json'}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()