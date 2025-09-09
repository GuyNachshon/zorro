"""
Malicious package data extraction from the encrypted dataset.
Handles both compromised_lib and malicious_intent categories.
"""

import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

@dataclass
class PackageSample:
    """Represents a malicious package sample."""
    name: str
    ecosystem: str  # npm, pypi
    category: str  # compromised_lib, malicious_intent
    version: Optional[str]
    file_path: Path
    extracted_path: Optional[Path] = None

class MaliciousExtractor:
    """Extracts and categorizes malicious packages from the encrypted dataset."""
    
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.password = "infected"
        
    def load_manifests(self) -> Dict[str, Dict[str, Optional[List[str]]]]:
        """Load and parse manifest.json files for both ecosystems."""
        manifests = {}
        
        for ecosystem in ["npm", "pypi"]:
            manifest_path = self.dataset_root / "samples" / ecosystem / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifests[ecosystem] = json.load(f)
            else:
                print(f"Warning: manifest for {ecosystem} not found at {manifest_path}")
                manifests[ecosystem] = {}
                
        return manifests
    
    def categorize_packages(self, manifests: Dict[str, Dict]) -> Dict[str, List[PackageSample]]:
        """Categorize packages into compromised_lib vs malicious_intent."""
        categories = {
            "compromised_lib": [],
            "malicious_intent": []
        }
        
        for ecosystem, manifest in manifests.items():
            for package_name, versions in manifest.items():
                if versions is None:
                    # null = malicious intent (fully malicious package)
                    category = "malicious_intent"
                    version_list = [None]  # Will find actual versions by scanning directories
                else:
                    # non-null = compromised library with specific bad versions
                    category = "compromised_lib"
                    version_list = versions
                
                # Find actual sample files for this package
                package_samples = self._find_package_files(ecosystem, package_name, category)
                
                for sample_path in package_samples:
                    sample = PackageSample(
                        name=package_name,
                        ecosystem=ecosystem,
                        category=category,
                        version=self._extract_version_from_path(sample_path),
                        file_path=sample_path
                    )
                    categories[category].append(sample)
        
        return categories
    
    def _find_package_files(self, ecosystem: str, package_name: str, category: str) -> List[Path]:
        """Find ZIP files for a given package in the dataset."""
        category_dir = self.dataset_root / "samples" / ecosystem / category
        
        # Look for package directories
        package_files = []
        if category_dir.exists():
            # Search for directories matching the package name
            package_dir = category_dir / package_name
            if package_dir.exists() and package_dir.is_dir():
                # Find ZIP files in package directory and subdirectories
                for zip_file in package_dir.rglob("*.zip"):
                    package_files.append(zip_file)
        
        return package_files
    
    def _extract_version_from_path(self, file_path: Path) -> Optional[str]:
        """Extract version from file path or filename."""
        # Try to extract version from filename pattern like "2023-03-20-package-v1.0.0.zip"
        filename = file_path.stem
        parts = filename.split('-')
        
        # Look for version-like patterns
        for i, part in enumerate(parts):
            if part.startswith('v') and len(part) > 1:
                return part[1:]  # Remove 'v' prefix
            elif part.replace('.', '').replace('_', '').isdigit():
                return part
        
        # Try parent directory name as version
        parent_name = file_path.parent.name
        if parent_name and parent_name != file_path.parent.parent.name:
            return parent_name
            
        return None
    
    def extract_sample(self, sample: PackageSample, output_dir: Optional[Path] = None) -> Path:
        """Extract a single encrypted sample to a directory."""
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix=f"icn_extract_{sample.ecosystem}_{sample.name}_"))
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(sample.file_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir, pwd=self.password.encode())
            
            sample.extracted_path = output_dir
            return output_dir
            
        except Exception as e:
            print(f"Error extracting {sample.file_path}: {e}")
            raise
    
    def get_sample_stats(self) -> Dict[str, int]:
        """Get statistics about the malicious dataset."""
        manifests = self.load_manifests()
        categories = self.categorize_packages(manifests)
        
        stats = {
            "total_packages": 0,
            "compromised_lib": len(categories["compromised_lib"]),
            "malicious_intent": len(categories["malicious_intent"]),
            "npm_samples": 0,
            "pypi_samples": 0
        }
        
        for category_samples in categories.values():
            for sample in category_samples:
                stats["total_packages"] += 1
                stats[f"{sample.ecosystem}_samples"] += 1
        
        return stats
    
    def extract_sample_batch(self, samples: List[PackageSample], 
                           output_base: Path, 
                           max_samples: Optional[int] = None) -> List[PackageSample]:
        """Extract multiple samples in batch."""
        if max_samples:
            samples = samples[:max_samples]
        
        extracted = []
        for i, sample in enumerate(samples):
            try:
                # Create organized directory structure
                sample_dir = output_base / sample.ecosystem / sample.category / sample.name
                if sample.version:
                    sample_dir = sample_dir / sample.version
                
                self.extract_sample(sample, sample_dir)
                extracted.append(sample)
                
                if (i + 1) % 100 == 0:
                    print(f"Extracted {i + 1}/{len(samples)} samples...")
                    
            except Exception as e:
                print(f"Failed to extract {sample.name}: {e}")
                continue
        
        return extracted

if __name__ == "__main__":
    # Example usage
    dataset_path = "/Users/guynachshon/Documents/baddon-ai/zorro/malicious-software-packages-dataset"
    extractor = MaliciousExtractor(dataset_path)
    
    # Get dataset statistics
    stats = extractor.get_sample_stats()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Load and categorize samples
    manifests = extractor.load_manifests()
    categories = extractor.categorize_packages(manifests)
    
    print(f"\nCategories:")
    print(f"  Compromised libraries: {len(categories['compromised_lib'])}")
    print(f"  Malicious intent: {len(categories['malicious_intent'])}")
    
    # Show some examples
    if categories["malicious_intent"]:
        print(f"\nFirst 5 malicious_intent samples:")
        for sample in categories["malicious_intent"][:5]:
            print(f"  {sample.ecosystem}/{sample.name} - {sample.file_path.name}")