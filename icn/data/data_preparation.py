"""
ICN Data Preparation Pipeline
Integrates malicious extraction with benign collection for complete training dataset.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import shutil

from .malicious_extractor import MaliciousExtractor, PackageSample
from .benign_collector import BenignCollector, BenignSample
from ..parsing.unified_parser import UnifiedParser, CodeUnit
from ..training.dataloader import ProcessedPackage
from ..training.losses import SampleType

logger = logging.getLogger(__name__)


class ICNDataPreparator:
    """Prepares complete ICN training dataset by processing malicious and benign samples."""
    
    def __init__(self,
                 malicious_dataset_path: str = "malicious-software-packages-dataset",
                 benign_cache_path: str = "data/benign_samples",
                 processed_cache_path: str = "data/processed_packages",
                 max_units_per_package: int = 50,
                 max_seq_length: int = 512,
                 parallel_workers: int = 4):
        
        self.malicious_dataset_path = Path(malicious_dataset_path)
        self.benign_cache_path = Path(benign_cache_path)
        self.processed_cache_path = Path(processed_cache_path)
        self.max_units_per_package = max_units_per_package
        self.max_seq_length = max_seq_length
        self.parallel_workers = parallel_workers
        
        # Create directories
        self.processed_cache_path.mkdir(parents=True, exist_ok=True)
        self.benign_cache_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.malicious_extractor = MaliciousExtractor(str(self.malicious_dataset_path))
        self.benign_collector = BenignCollector()
        self.parser = UnifiedParser()
        
        logger.info(f"🔧 ICN Data Preparator initialized")
        logger.info(f"   Malicious dataset: {self.malicious_dataset_path}")
        logger.info(f"   Benign cache: {self.benign_cache_path}")
        logger.info(f"   Processed cache: {self.processed_cache_path}")
    
    def get_malicious_samples(self, max_samples: Optional[int] = None) -> Dict[str, List[PackageSample]]:
        """Get categorized malicious samples from the dataset."""
        logger.info("🔍 Loading malicious samples...")
        
        # Load manifests and categorize
        manifests = self.malicious_extractor.load_manifests()
        categories = self.malicious_extractor.categorize_packages(manifests)
        
        if max_samples:
            # Limit samples for testing
            for category in categories:
                categories[category] = categories[category][:max_samples//2]
        
        logger.info(f"📊 Malicious samples loaded:")
        logger.info(f"   compromised_lib: {len(categories['compromised_lib'])}")
        logger.info(f"   malicious_intent: {len(categories['malicious_intent'])}")
        
        return categories
    
    def get_benign_samples(self, target_count: int = 5000) -> List[BenignSample]:
        """Collect benign samples for training."""
        logger.info(f"🔍 Collecting {target_count} benign samples...")
        
        # Try to load from cache first
        cache_file = self.benign_cache_path / "benign_samples.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                benign_samples = []
                for sample_data in cached_data:
                    sample = BenignSample(
                        name=sample_data['name'],
                        ecosystem=sample_data['ecosystem'],
                        version=sample_data.get('version'),
                        source_url=sample_data.get('source_url'),
                        local_path=Path(sample_data['local_path']) if sample_data.get('local_path') else None
                    )
                    benign_samples.append(sample)
                
                if len(benign_samples) >= target_count:
                    logger.info(f"✅ Loaded {len(benign_samples)} benign samples from cache")
                    return benign_samples[:target_count]
                    
            except Exception as e:
                logger.warning(f"Failed to load benign cache: {e}")
        
        # Collect new benign samples
        benign_samples = self.benign_collector.collect_popular_packages(
            target_count=target_count,
            cache_dir=self.benign_cache_path
        )
        
        # Save to cache
        try:
            cache_data = []
            for sample in benign_samples:
                cache_data.append({
                    'name': sample.name,
                    'ecosystem': sample.ecosystem,
                    'version': sample.version,
                    'source_url': sample.source_url,
                    'local_path': str(sample.local_path) if sample.local_path else None
                })
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            logger.info(f"💾 Cached {len(benign_samples)} benign samples")
            
        except Exception as e:
            logger.warning(f"Failed to cache benign samples: {e}")
        
        return benign_samples
    
    def process_single_malicious_sample(self, sample: PackageSample) -> Optional[ProcessedPackage]:
        """Process a single malicious sample into a ProcessedPackage."""
        try:
            # Extract sample to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract the ZIP file
                extracted_path = self.malicious_extractor.extract_sample(sample, temp_path)
                
                # Parse the extracted package
                code_units = self.parser.parse_package(extracted_path)
                
                if not code_units or len(code_units) == 0:
                    logger.warning(f"No code units found in {sample.name}")
                    return None
                
                # Limit units per package
                if len(code_units) > self.max_units_per_package:
                    code_units = code_units[:self.max_units_per_package]
                
                # Convert category to SampleType
                sample_type = SampleType.COMPROMISED_LIB if sample.category == "compromised_lib" else SampleType.MALICIOUS_INTENT
                
                # Create processed package
                processed = ProcessedPackage(
                    name=sample.name,
                    ecosystem=sample.ecosystem,
                    sample_type=sample_type,
                    units=code_units,
                    malicious_label=1.0,  # All malicious samples get label 1
                    package_hash=f"{sample.ecosystem}_{sample.name}_{sample.version or 'unknown'}"
                )
                
                # Compute features using parser
                self._compute_package_features(processed)
                
                return processed
                
        except Exception as e:
            logger.error(f"Failed to process malicious sample {sample.name}: {e}")
            return None
    
    def process_single_benign_sample(self, sample: BenignSample) -> Optional[ProcessedPackage]:
        """Process a single benign sample into a ProcessedPackage."""
        try:
            if not sample.local_path or not sample.local_path.exists():
                logger.warning(f"Benign sample {sample.name} not found at {sample.local_path}")
                return None
            
            # Parse the package
            code_units = self.parser.parse_package(sample.local_path)
            
            if not code_units or len(code_units) == 0:
                logger.warning(f"No code units found in {sample.name}")
                return None
            
            # Limit units per package
            if len(code_units) > self.max_units_per_package:
                code_units = code_units[:self.max_units_per_package]
            
            # Create processed package
            processed = ProcessedPackage(
                name=sample.name,
                ecosystem=sample.ecosystem,
                sample_type=SampleType.BENIGN,
                units=code_units,
                malicious_label=0.0,  # All benign samples get label 0
                package_hash=f"{sample.ecosystem}_{sample.name}_{sample.version or 'unknown'}"
            )
            
            # Compute features using parser
            self._compute_package_features(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"Failed to process benign sample {sample.name}: {e}")
            return None
    
    def _compute_package_features(self, package: ProcessedPackage):
        """Compute features for a processed package."""
        n_units = len(package.units)
        
        if n_units == 0:
            return
        
        # Initialize tensors
        input_ids_list = []
        attention_masks_list = []
        phase_ids_list = []
        api_features_list = []
        ast_features_list = []
        
        for unit in package.units:
            # Use parser to compute features for each unit
            features = self.parser.compute_unit_features(unit, max_seq_length=self.max_seq_length)
            
            input_ids_list.append(features['input_ids'])
            attention_masks_list.append(features['attention_mask'])
            phase_ids_list.append(features.get('phase_id', 0))  # Default to 0 if not available
            api_features_list.append(features.get('api_features', [0.0] * 15))  # Default API features
            ast_features_list.append(features.get('ast_features', [0.0] * 10))  # Default AST features
        
        # Stack into tensors
        package.input_ids = torch.stack(input_ids_list)
        package.attention_masks = torch.stack(attention_masks_list)
        package.phase_ids = torch.tensor(phase_ids_list, dtype=torch.long)
        package.api_features = torch.tensor(api_features_list, dtype=torch.float)
        package.ast_features = torch.tensor(ast_features_list, dtype=torch.float)
    
    def prepare_complete_dataset(self, 
                               max_malicious_samples: Optional[int] = None,
                               target_benign_count: int = 5000,
                               force_recompute: bool = False) -> List[ProcessedPackage]:
        """
        Prepare the complete ICN training dataset.
        
        Args:
            max_malicious_samples: Maximum malicious samples to process (for testing)
            target_benign_count: Target number of benign samples
            force_recompute: Force recomputation even if cache exists
            
        Returns:
            List of processed packages ready for training
        """
        logger.info("🚀 Preparing complete ICN dataset...")
        
        # Check for cached processed packages
        cache_file = self.processed_cache_path / "processed_packages.json"
        if cache_file.exists() and not force_recompute:
            logger.info("📦 Loading processed packages from cache...")
            try:
                # Load cached packages (implementation would depend on serialization format)
                # For now, we'll always recompute
                pass
            except Exception as e:
                logger.warning(f"Failed to load cached packages: {e}")
        
        processed_packages = []
        
        # Process malicious samples
        logger.info("🦠 Processing malicious samples...")
        malicious_samples = self.get_malicious_samples(max_malicious_samples)
        
        all_malicious = []
        for category, samples in malicious_samples.items():
            all_malicious.extend(samples)
        
        logger.info(f"Processing {len(all_malicious)} malicious samples...")
        
        # Process malicious samples with parallel processing
        successful_malicious = 0
        with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
            future_to_sample = {
                executor.submit(self.process_single_malicious_sample, sample): sample
                for sample in all_malicious
            }
            
            for future in as_completed(future_to_sample):
                processed = future.result()
                if processed:
                    processed_packages.append(processed)
                    successful_malicious += 1
                    
                    if successful_malicious % 100 == 0:
                        logger.info(f"✅ Processed {successful_malicious}/{len(all_malicious)} malicious samples")
        
        logger.info(f"🦠 Completed malicious processing: {successful_malicious}/{len(all_malicious)} successful")
        
        # Process benign samples
        logger.info("🟢 Processing benign samples...")
        benign_samples = self.get_benign_samples(target_benign_count)
        
        successful_benign = 0
        with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
            future_to_sample = {
                executor.submit(self.process_single_benign_sample, sample): sample
                for sample in benign_samples
            }
            
            for future in as_completed(future_to_sample):
                processed = future.result()
                if processed:
                    processed_packages.append(processed)
                    successful_benign += 1
                    
                    if successful_benign % 100 == 0:
                        logger.info(f"✅ Processed {successful_benign}/{len(benign_samples)} benign samples")
        
        logger.info(f"🟢 Completed benign processing: {successful_benign}/{len(benign_samples)} successful")
        
        # Final statistics
        total_processed = len(processed_packages)
        logger.info(f"📊 Dataset preparation complete:")
        logger.info(f"   Total packages: {total_processed}")
        logger.info(f"   Malicious: {successful_malicious}")
        logger.info(f"   Benign: {successful_benign}")
        
        # Count by category
        category_stats = {}
        for pkg in processed_packages:
            category = pkg.sample_type.value
            category_stats[category] = category_stats.get(category, 0) + 1
        
        for category, count in category_stats.items():
            logger.info(f"   {category}: {count} ({count/total_processed*100:.1f}%)")
        
        return processed_packages
    
    def save_dataset_statistics(self, processed_packages: List[ProcessedPackage]):
        """Save comprehensive dataset statistics."""
        stats = {
            'total_packages': len(processed_packages),
            'categories': {},
            'ecosystems': {},
            'average_units_per_package': 0,
            'total_code_units': 0
        }
        
        total_units = 0
        for pkg in processed_packages:
            # Category stats
            category = pkg.sample_type.value
            stats['categories'][category] = stats['categories'].get(category, 0) + 1
            
            # Ecosystem stats
            ecosystem = pkg.ecosystem
            stats['ecosystems'][ecosystem] = stats['ecosystems'].get(ecosystem, 0) + 1
            
            # Unit counts
            n_units = len(pkg.units)
            total_units += n_units
        
        stats['total_code_units'] = total_units
        stats['average_units_per_package'] = total_units / len(processed_packages) if processed_packages else 0
        
        # Save to file
        stats_file = self.processed_cache_path / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"📊 Dataset statistics saved to {stats_file}")


# Import torch at module level to avoid issues
import torch


if __name__ == "__main__":
    # Test data preparation
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    preparator = ICNDataPreparator()
    
    # Prepare a small test dataset
    test_packages = preparator.prepare_complete_dataset(
        max_malicious_samples=100,  # Small test set
        target_benign_count=200,
        force_recompute=True
    )
    
    print(f"\n✅ Test dataset prepared: {len(test_packages)} packages")
    
    # Save statistics
    preparator.save_dataset_statistics(test_packages)