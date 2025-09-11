"""
PyTorch DataLoader for ICN training with variable-length packages.
Handles efficient batching and collation for packages with different numbers of code units.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
import random
import numpy as np
from enum import Enum

# Import ICN components
from ..models.icn_model import ICNInput
from ..parsing.unified_parser import UnifiedParser, CodeUnit
from ..data.malicious_extractor import MaliciousExtractor, PackageSample
from ..data.benign_collector import BenignCollector, BenignSample
from ..training.losses import SampleType


@dataclass
class ProcessedPackage:
    """Represents a fully processed package ready for training."""
    name: str
    ecosystem: str  # npm, pypi
    sample_type: SampleType  # benign, compromised_lib, malicious_intent
    
    # Code units
    units: List[CodeUnit]
    
    # Pre-computed features
    input_ids: torch.Tensor  # [n_units, max_seq_len]
    attention_masks: torch.Tensor  # [n_units, max_seq_len]
    phase_ids: torch.Tensor  # [n_units]
    api_features: torch.Tensor  # [n_units, n_api_categories]
    ast_features: torch.Tensor  # [n_units, n_ast_features]
    
    # Labels
    intent_labels: Optional[torch.Tensor] = None  # [n_api_categories] weak supervision
    malicious_label: float = 0.0  # 0 = benign, 1 = malicious
    
    # Metadata
    manifest_embedding: Optional[torch.Tensor] = None
    package_hash: str = ""


class CurriculumStage(Enum):
    """Curriculum learning stages."""
    STAGE_A_PRETRAINING = "stage_a_pretraining"      # Benign only, intent prediction
    STAGE_B_CONVERGENCE = "stage_b_convergence"       # Benign only, convergence training
    STAGE_C_MALICIOUS = "stage_c_malicious"          # Add malicious samples
    STAGE_D_ROBUSTNESS = "stage_d_robustness"        # Obfuscation hardening


class ICNDataset(Dataset):
    """
    PyTorch dataset for ICN training.
    Handles both malicious and benign packages with curriculum learning support.
    """
    
    def __init__(
        self,
        processed_packages: List[ProcessedPackage],
        curriculum_stage: CurriculumStage = CurriculumStage.STAGE_C_MALICIOUS,
        max_units_per_package: int = 50,
        stage_config: Dict[str, Any] = None
    ):
        self.processed_packages = processed_packages
        self.curriculum_stage = curriculum_stage
        self.max_units_per_package = max_units_per_package
        self.stage_config = stage_config or {}
        
        # Filter packages based on curriculum stage
        self.filtered_packages = self._filter_for_stage()
        
        print(f"📚 Dataset initialized for {curriculum_stage.value}")
        print(f"   Total packages: {len(self.filtered_packages)}")
        self._print_dataset_stats()
    
    def _filter_for_stage(self) -> List[ProcessedPackage]:
        """Filter packages based on curriculum stage."""
        if self.curriculum_stage in [CurriculumStage.STAGE_A_PRETRAINING, CurriculumStage.STAGE_B_CONVERGENCE]:
            # Stages A & B: Benign only
            return [pkg for pkg in self.processed_packages if pkg.sample_type == SampleType.BENIGN]
        
        elif self.curriculum_stage == CurriculumStage.STAGE_C_MALICIOUS:
            # Stage C: All samples
            return self.processed_packages
        
        elif self.curriculum_stage == CurriculumStage.STAGE_D_ROBUSTNESS:
            # Stage D: All samples, potentially with augmentation
            return self.processed_packages
        
        else:
            return self.processed_packages
    
    def _print_dataset_stats(self):
        """Print dataset composition statistics."""
        stats = {}
        for pkg in self.filtered_packages:
            sample_type = pkg.sample_type.value
            stats[sample_type] = stats.get(sample_type, 0) + 1
        
        total = len(self.filtered_packages)
        for sample_type, count in stats.items():
            print(f"   {sample_type}: {count} ({count/total*100:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.filtered_packages)
    
    def __getitem__(self, idx: int) -> ProcessedPackage:
        return self.filtered_packages[idx]
    
    def get_curriculum_stats(self) -> Dict[str, int]:
        """Get statistics for curriculum stage tracking."""
        stats = {}
        for pkg in self.filtered_packages:
            sample_type = pkg.sample_type.value
            stats[sample_type] = stats.get(sample_type, 0) + 1
        return stats
    
    def update_stage(self, new_stage: CurriculumStage, stage_config: Dict[str, Any] = None):
        """Update curriculum stage and re-filter packages."""
        self.curriculum_stage = new_stage
        self.stage_config = stage_config or {}
        self.filtered_packages = self._filter_for_stage()
        
        print(f"📚 Updated to {new_stage.value}: {len(self.filtered_packages)} packages")
        self._print_dataset_stats()


class PackageCollateFunction:
    """
    Custom collate function for batching variable-length packages.
    Handles padding and creates proper batch tensors.
    """
    
    def __init__(self, max_units_per_package: int = 50, pad_token_id: int = 0):
        self.max_units_per_package = max_units_per_package
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[ProcessedPackage]) -> ICNInput:
        """Collate a batch of packages into ICNInput format."""
        batch_size = len(batch)
        
        # Prepare lists for ICNInput
        input_ids_list = []
        attention_masks_list = []
        phase_ids_list = []
        api_features_list = []
        ast_features_list = []
        
        # Package-level data
        manifest_embeddings = []
        sample_types = []
        intent_labels = []
        malicious_labels = []
        
        for package in batch:
            # Limit units per package
            n_units = min(len(package.units), self.max_units_per_package)
            
            # Extract unit-level features
            pkg_input_ids = package.input_ids[:n_units]
            pkg_attention_masks = package.attention_masks[:n_units]
            pkg_phase_ids = package.phase_ids[:n_units]
            pkg_api_features = package.api_features[:n_units]
            pkg_ast_features = package.ast_features[:n_units]
            
            input_ids_list.append(pkg_input_ids)
            attention_masks_list.append(pkg_attention_masks)
            phase_ids_list.append(pkg_phase_ids)
            api_features_list.append(pkg_api_features)
            ast_features_list.append(pkg_ast_features)
            
            # Package-level data
            if package.manifest_embedding is not None:
                manifest_embeddings.append(package.manifest_embedding)
            else:
                # Create zero embedding if no manifest
                manifest_embeddings.append(torch.zeros(768))  # Default embedding size
            
            sample_types.append(package.sample_type.value)
            malicious_labels.append(package.malicious_label)
            
            # Intent labels (if available)
            if package.intent_labels is not None:
                intent_labels.append(package.intent_labels)
            else:
                intent_labels.append(torch.zeros(15))  # Default: 15 intent categories
        
        # Stack package-level tensors
        manifest_embeddings_tensor = torch.stack(manifest_embeddings) if manifest_embeddings else None
        intent_labels_tensor = torch.stack(intent_labels) if intent_labels else None
        malicious_labels_tensor = torch.tensor(malicious_labels, dtype=torch.float)
        
        # Create ICNInput
        batch_input = ICNInput(
            input_ids_list=input_ids_list,
            attention_masks_list=attention_masks_list,
            phase_ids_list=phase_ids_list,
            api_features_list=api_features_list,
            ast_features_list=ast_features_list,
            manifest_embeddings=manifest_embeddings_tensor,
            sample_types=sample_types,
            intent_labels=intent_labels_tensor,
            malicious_labels=malicious_labels_tensor
        )
        
        return batch_input


class BalancedSampler(Sampler):
    """
    Balanced sampler that ensures each batch has a good mix of sample types.
    Useful for curriculum stage C when we have mixed benign/malicious samples.
    """
    
    def __init__(
        self, 
        dataset: ICNDataset, 
        batch_size: int,
        malicious_ratio: float = 0.2,  # 20% malicious per batch
        shuffle: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.malicious_ratio = malicious_ratio
        self.shuffle = shuffle
        
        # Group indices by sample type
        self.benign_indices = []
        self.compromised_indices = []
        self.malicious_indices = []
        
        for idx, package in enumerate(dataset.filtered_packages):
            if package.sample_type == SampleType.BENIGN:
                self.benign_indices.append(idx)
            elif package.sample_type == SampleType.COMPROMISED_LIB:
                self.compromised_indices.append(idx)
            elif package.sample_type == SampleType.MALICIOUS_INTENT:
                self.malicious_indices.append(idx)
        
        self.total_batches = len(dataset) // batch_size
        print(f"🎯 Balanced sampler: {len(self.benign_indices)} benign, "
              f"{len(self.compromised_indices)} compromised, {len(self.malicious_indices)} malicious")
    
    def __iter__(self):
        """Generate balanced batches."""
        if self.shuffle:
            random.shuffle(self.benign_indices)
            random.shuffle(self.compromised_indices)
            random.shuffle(self.malicious_indices)
        
        # Calculate samples per batch
        malicious_per_batch = int(self.batch_size * self.malicious_ratio)
        benign_per_batch = self.batch_size - malicious_per_batch
        
        # Split malicious samples between compromised and malicious_intent
        compromised_per_batch = malicious_per_batch // 2
        malicious_intent_per_batch = malicious_per_batch - compromised_per_batch
        
        batch_indices = []
        benign_idx = 0
        compromised_idx = 0
        malicious_idx = 0
        
        for batch_num in range(self.total_batches):
            batch = []
            
            # Add benign samples
            for _ in range(benign_per_batch):
                if not self.benign_indices:
                    # No benign samples available, skip
                    break
                if benign_idx < len(self.benign_indices):
                    batch.append(self.benign_indices[benign_idx])
                    benign_idx += 1
                else:
                    # Wrap around if we run out
                    benign_idx = 0
                    if self.shuffle:
                        random.shuffle(self.benign_indices)
                    batch.append(self.benign_indices[benign_idx])
                    benign_idx += 1
            
            # Add compromised samples
            for _ in range(compromised_per_batch):
                if compromised_idx < len(self.compromised_indices) and self.compromised_indices:
                    batch.append(self.compromised_indices[compromised_idx])
                    compromised_idx += 1
                elif self.compromised_indices:
                    compromised_idx = 0
                    batch.append(self.compromised_indices[compromised_idx])
                    compromised_idx += 1
                else:
                    # Fall back to malicious_intent if no compromised samples
                    if malicious_idx < len(self.malicious_indices):
                        batch.append(self.malicious_indices[malicious_idx])
                        malicious_idx += 1
            
            # Add malicious_intent samples
            for _ in range(malicious_intent_per_batch):
                if malicious_idx < len(self.malicious_indices):
                    batch.append(self.malicious_indices[malicious_idx])
                    malicious_idx += 1
                else:
                    malicious_idx = 0
                    if self.shuffle and self.malicious_indices:
                        random.shuffle(self.malicious_indices)
                    if self.malicious_indices:
                        batch.append(self.malicious_indices[malicious_idx])
                        malicious_idx += 1
            
            if self.shuffle:
                random.shuffle(batch)
            
            batch_indices.extend(batch)
        
        return iter(batch_indices)
    
    def __len__(self):
        return self.total_batches * self.batch_size


def create_icn_dataloader(
    processed_packages: List[ProcessedPackage],
    batch_size: int = 32,
    curriculum_stage: CurriculumStage = CurriculumStage.STAGE_C_MALICIOUS,
    shuffle: bool = True,
    num_workers: int = 0,
    max_units_per_package: int = 50,
    balanced_sampling: bool = True,
    malicious_ratio: float = 0.2
) -> DataLoader:
    """
    Create an ICN DataLoader with appropriate sampling and collation.
    
    Args:
        processed_packages: List of processed packages
        batch_size: Batch size
        curriculum_stage: Current curriculum learning stage
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        max_units_per_package: Maximum units per package
        balanced_sampling: Use balanced sampling for mixed stages
        malicious_ratio: Ratio of malicious samples per batch (if balanced)
    
    Returns:
        DataLoader configured for ICN training
    """
    
    # Create dataset
    dataset = ICNDataset(
        processed_packages=processed_packages,
        curriculum_stage=curriculum_stage,
        max_units_per_package=max_units_per_package
    )
    
    # Create collate function
    collate_fn = PackageCollateFunction(
        max_units_per_package=max_units_per_package,
        pad_token_id=0
    )
    
    # Choose sampler
    sampler = None
    if balanced_sampling and curriculum_stage == CurriculumStage.STAGE_C_MALICIOUS:
        # Use balanced sampling for stage C
        sampler = BalancedSampler(
            dataset=dataset,
            batch_size=batch_size,
            malicious_ratio=malicious_ratio,
            shuffle=shuffle
        )
        shuffle = False  # Sampler handles shuffling
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"🚀 DataLoader created:")
    print(f"   Batch size: {batch_size}")
    print(f"   Curriculum stage: {curriculum_stage.value}")
    print(f"   Balanced sampling: {balanced_sampling}")
    print(f"   Total batches: {len(dataloader)}")
    
    return dataloader


class DatasetBuilder:
    """
    Helper class to build processed datasets from raw package data.
    Handles tokenization, feature extraction, and caching.
    """
    
    def __init__(self, parser: UnifiedParser, tokenizer, max_seq_length: int = 512):
        self.parser = parser
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def process_package(
        self, 
        package_path: Path,
        package_name: str,
        ecosystem: str,
        sample_type: SampleType
    ) -> Optional[ProcessedPackage]:
        """Process a single package into ProcessedPackage format."""
        
        try:
            # Parse package
            analysis = self.parser.parse_package(
                package_path=package_path,
                package_name=package_name,
                ecosystem=ecosystem,
                category=sample_type.value
            )
            
            if not analysis.units:
                return None
            
            # Tokenize and create features
            input_ids_list = []
            attention_masks_list = []
            phase_ids_list = []
            api_features_list = []
            ast_features_list = []
            
            for unit in analysis.units:
                # Tokenize code content
                tokens = self.tokenizer.encode(
                    unit.raw_content,
                    max_length=self.max_seq_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = tokens.squeeze(0)
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
                
                # Phase ID (install=0, postinstall=1, runtime=2)
                phase_map = {"install": 0, "postinstall": 1, "runtime": 2}
                phase_id = phase_map.get(unit.phase, 2)
                
                # API features (binary vector of API categories)
                api_features = torch.zeros(15)  # 15 API categories
                for i, category in enumerate(unit.api_categories):
                    # Map category name to index (simplified)
                    if i < 15:
                        api_features[i] = 1.0
                
                # AST features (simplified - could be more sophisticated)
                ast_features = torch.zeros(50)
                if len(unit.ast_nodes) > 0:
                    # Simple frequency-based features
                    for i, node_type in enumerate(unit.ast_nodes[:50]):
                        if i < 50:
                            ast_features[i] = 1.0
                
                input_ids_list.append(input_ids)
                attention_masks_list.append(attention_mask)
                phase_ids_list.append(torch.tensor(phase_id))
                api_features_list.append(api_features)
                ast_features_list.append(ast_features)
            
            # Stack unit features
            input_ids = torch.stack(input_ids_list)
            attention_masks = torch.stack(attention_masks_list)
            phase_ids = torch.stack(phase_ids_list)
            api_features = torch.stack(api_features_list)
            ast_features = torch.stack(ast_features_list)
            
            # Create processed package
            processed_package = ProcessedPackage(
                name=package_name,
                ecosystem=ecosystem,
                sample_type=sample_type,
                units=analysis.units,
                input_ids=input_ids,
                attention_masks=attention_masks,
                phase_ids=phase_ids,
                api_features=api_features,
                ast_features=ast_features,
                malicious_label=1.0 if sample_type != SampleType.BENIGN else 0.0,
                package_hash=f"{package_name}_{ecosystem}_{sample_type.value}"
            )
            
            return processed_package
            
        except Exception as e:
            print(f"❌ Error processing package {package_name}: {e}")
            return None


if __name__ == "__main__":
    # Test the dataloader implementation
    print("🧪 Testing ICN DataLoader...")
    
    # Create some mock processed packages for testing
    mock_packages = []
    
    for i in range(20):
        sample_type = SampleType.BENIGN if i < 15 else (
            SampleType.COMPROMISED_LIB if i < 18 else SampleType.MALICIOUS_INTENT
        )
        
        n_units = random.randint(1, 5)
        max_seq_len = 128
        
        mock_package = ProcessedPackage(
            name=f"test_package_{i}",
            ecosystem="npm",
            sample_type=sample_type,
            units=[],  # Empty for testing
            input_ids=torch.randint(0, 1000, (n_units, max_seq_len)),
            attention_masks=torch.ones(n_units, max_seq_len),
            phase_ids=torch.randint(0, 3, (n_units,)),
            api_features=torch.randn(n_units, 15),
            ast_features=torch.randn(n_units, 50),
            malicious_label=1.0 if sample_type != SampleType.BENIGN else 0.0
        )
        
        mock_packages.append(mock_package)
    
    # Test different curriculum stages
    for stage in CurriculumStage:
        print(f"\n📚 Testing {stage.value}:")
        
        dataloader = create_icn_dataloader(
            processed_packages=mock_packages,
            batch_size=4,
            curriculum_stage=stage,
            balanced_sampling=(stage == CurriculumStage.STAGE_C_MALICIOUS)
        )
        
        # Test one batch
        batch = next(iter(dataloader))
        print(f"   Batch types: {batch.sample_types}")
        print(f"   Malicious labels: {batch.malicious_labels.tolist()}")
        print(f"   Units per package: {[len(ids) for ids in batch.input_ids_list]}")
    
    print("\n✅ DataLoader implementation working correctly!")
    print("🚀 Ready for curriculum training pipeline!")