#!/usr/bin/env python3
"""
Comprehensive extraction and processing of all malicious package samples.
Prepares the complete dataset for ICN training pipeline.
"""

import os
import json
import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import argparse
import tempfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sample_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MaliciousSampleExtractor:
    """Extracts and processes malicious package samples for ICN training."""
    
    def __init__(self, 
                 samples_dir: str = "malicious-software-packages-dataset/samples",
                 output_dir: str = "data/extracted_samples",
                 max_workers: int = 4):
        self.samples_dir = Path(samples_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "npm").mkdir(exist_ok=True)
        (self.output_dir / "pypi").mkdir(exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'npm': {'malicious_intent': 0, 'compromised_lib': 0, 'total_files': 0, 'failed': 0},
            'pypi': {'malicious_intent': 0, 'compromised_lib': 0, 'total_files': 0, 'failed': 0},
            'extraction_errors': []
        }
    
    def discover_samples(self) -> Dict[str, Dict[str, List[Tuple[str, str, str]]]]:
        """
        Discover all malicious sample ZIP files.
        
        Returns:
            Dict with structure: {ecosystem: {category: [(package_name, version, zip_path)]}}
        """
        logger.info("🔍 Discovering malicious samples...")
        
        samples = {'npm': {'malicious_intent': [], 'compromised_lib': []},
                  'pypi': {'malicious_intent': [], 'compromised_lib': []}}
        
        for ecosystem in ['npm', 'pypi']:
            ecosystem_path = self.samples_dir / ecosystem
            if not ecosystem_path.exists():
                logger.warning(f"Ecosystem path not found: {ecosystem_path}")
                continue
                
            for category in ['malicious_intent', 'compromised_lib']:
                category_path = ecosystem_path / category
                if not category_path.exists():
                    logger.warning(f"Category path not found: {category_path}")
                    continue
                
                # Walk through package directories
                for package_dir in category_path.iterdir():
                    if not package_dir.is_dir():
                        continue
                    
                    package_name = package_dir.name
                    
                    # Walk through version directories
                    for version_dir in package_dir.iterdir():
                        if not version_dir.is_dir():
                            continue
                        
                        version = version_dir.name
                        
                        # Find ZIP files in version directory
                        for zip_file in version_dir.glob("*.zip"):
                            samples[ecosystem][category].append((package_name, version, str(zip_file)))
        
        # Log discovery statistics
        total_samples = 0
        for ecosystem, categories in samples.items():
            ecosystem_total = sum(len(category_samples) for category_samples in categories.values())
            total_samples += ecosystem_total
            logger.info(f"📦 {ecosystem.upper()}: {ecosystem_total} samples")
            for category, category_samples in categories.items():
                logger.info(f"   {category}: {len(category_samples)} samples")
        
        logger.info(f"🎯 Total discovered: {total_samples} malicious samples")
        return samples
    
    def extract_single_package(self, args: Tuple[str, str, str, str, str]) -> Dict[str, Any]:
        """
        Extract a single package ZIP file.
        
        Args:
            args: (ecosystem, category, package_name, version, zip_path)
            
        Returns:
            Extraction result with metadata
        """
        ecosystem, category, package_name, version, zip_path = args
        
        result = {
            'ecosystem': ecosystem,
            'category': category,
            'package_name': package_name,
            'version': version,
            'zip_path': zip_path,
            'success': False,
            'extracted_files': [],
            'file_count': 0,
            'total_size': 0,
            'error': None
        }
        
        try:
            # Create output directory for this package
            package_output_dir = (self.output_dir / ecosystem / category / 
                                package_name / version)
            package_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get file list
                file_list = zip_ref.namelist()
                
                # Extract all files
                with tempfile.TemporaryDirectory() as temp_dir:
                    zip_ref.extractall(temp_dir)
                    
                    # Process extracted files
                    temp_path = Path(temp_dir)
                    extracted_files = []
                    total_size = 0
                    
                    for extracted_file in temp_path.rglob("*"):
                        if extracted_file.is_file():
                            # Calculate relative path from temp directory
                            rel_path = extracted_file.relative_to(temp_path)
                            
                            # Determine output path
                            output_file_path = package_output_dir / rel_path
                            output_file_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            # Copy file to output directory
                            shutil.copy2(extracted_file, output_file_path)
                            
                            file_size = output_file_path.stat().st_size
                            total_size += file_size
                            
                            extracted_files.append({
                                'path': str(rel_path),
                                'size': file_size,
                                'extension': extracted_file.suffix.lower()
                            })
            
            result.update({
                'success': True,
                'extracted_files': extracted_files,
                'file_count': len(extracted_files),
                'total_size': total_size
            })
            
            # Log successful extraction
            logger.debug(f"✅ Extracted {package_name} v{version} ({len(extracted_files)} files, {total_size:,} bytes)")
            
        except Exception as e:
            error_msg = f"Failed to extract {package_name} v{version}: {str(e)}"
            result['error'] = error_msg
            logger.error(f"❌ {error_msg}")
        
        return result
    
    def extract_all_samples(self, samples: Dict[str, Dict[str, List[Tuple[str, str, str]]]]) -> Dict[str, Any]:
        """
        Extract all discovered samples using parallel processing.
        
        Args:
            samples: Discovered samples structure
            
        Returns:
            Comprehensive extraction results
        """
        logger.info("🚀 Starting parallel extraction of all samples...")
        
        # Prepare extraction tasks
        extraction_tasks = []
        for ecosystem, categories in samples.items():
            for category, category_samples in categories.items():
                for package_name, version, zip_path in category_samples:
                    extraction_tasks.append((ecosystem, category, package_name, version, zip_path))
        
        total_tasks = len(extraction_tasks)
        logger.info(f"📋 Queued {total_tasks} extraction tasks")
        
        # Process extractions in parallel
        results = []
        completed = 0
        failed = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.extract_single_package, task): task 
                for task in extraction_tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)
                
                completed += 1
                if result['success']:
                    # Update statistics
                    ecosystem = result['ecosystem']
                    category = result['category']
                    self.stats[ecosystem][category] += 1
                    self.stats[ecosystem]['total_files'] += result['file_count']
                else:
                    failed += 1
                    self.stats[result['ecosystem']]['failed'] += 1
                    self.stats['extraction_errors'].append({
                        'package': f"{result['package_name']} v{result['version']}",
                        'ecosystem': result['ecosystem'],
                        'error': result['error']
                    })
                
                # Progress reporting
                if completed % 100 == 0:
                    progress = (completed / total_tasks) * 100
                    logger.info(f"📈 Progress: {completed}/{total_tasks} ({progress:.1f}%) - Failed: {failed}")
        
        # Final statistics
        logger.info(f"✅ Extraction complete: {completed - failed}/{total_tasks} successful")
        logger.info(f"❌ Failed extractions: {failed}")
        
        return {
            'total_tasks': total_tasks,
            'successful': completed - failed,
            'failed': failed,
            'results': results,
            'statistics': self.stats
        }
    
    def generate_dataset_manifest(self, extraction_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive manifest of the extracted dataset.
        
        Args:
            extraction_results: Results from extraction process
            
        Returns:
            Dataset manifest
        """
        logger.info("📋 Generating dataset manifest...")
        
        manifest = {
            'extraction_date': datetime.now().isoformat(),
            'total_packages': extraction_results['successful'],
            'failed_packages': extraction_results['failed'],
            'ecosystems': {},
            'file_type_distribution': {},
            'size_statistics': {
                'total_size_bytes': 0,
                'average_package_size': 0,
                'largest_package': None,
                'smallest_package': None
            },
            'categories': {
                'malicious_intent': 0,
                'compromised_lib': 0
            }
        }
        
        # Process successful extractions
        file_extensions = {}
        package_sizes = []
        largest_package = {'size': 0, 'name': ''}
        smallest_package = {'size': float('inf'), 'name': ''}
        
        for result in extraction_results['results']:
            if not result['success']:
                continue
                
            ecosystem = result['ecosystem']
            category = result['category']
            package_name = result['package_name']
            version = result['version']
            total_size = result['total_size']
            
            # Initialize ecosystem if not exists
            if ecosystem not in manifest['ecosystems']:
                manifest['ecosystems'][ecosystem] = {
                    'total_packages': 0,
                    'categories': {'malicious_intent': 0, 'compromised_lib': 0},
                    'total_files': 0,
                    'total_size_bytes': 0
                }
            
            # Update ecosystem statistics
            manifest['ecosystems'][ecosystem]['total_packages'] += 1
            manifest['ecosystems'][ecosystem]['categories'][category] += 1
            manifest['ecosystems'][ecosystem]['total_files'] += result['file_count']
            manifest['ecosystems'][ecosystem]['total_size_bytes'] += total_size
            
            # Update category statistics
            manifest['categories'][category] += 1
            
            # Process file extensions
            for file_info in result['extracted_files']:
                ext = file_info['extension'] or 'no_extension'
                file_extensions[ext] = file_extensions.get(ext, 0) + 1
            
            # Track package sizes
            package_sizes.append(total_size)
            manifest['size_statistics']['total_size_bytes'] += total_size
            
            # Track largest and smallest packages
            if total_size > largest_package['size']:
                largest_package = {
                    'size': total_size,
                    'name': f"{package_name} v{version} ({ecosystem})"
                }
            
            if total_size < smallest_package['size']:
                smallest_package = {
                    'size': total_size,
                    'name': f"{package_name} v{version} ({ecosystem})"
                }
        
        # Finalize size statistics
        if package_sizes:
            manifest['size_statistics']['average_package_size'] = sum(package_sizes) / len(package_sizes)
            manifest['size_statistics']['largest_package'] = largest_package
            manifest['size_statistics']['smallest_package'] = smallest_package
        
        # File type distribution
        manifest['file_type_distribution'] = dict(sorted(file_extensions.items(), 
                                                        key=lambda x: x[1], reverse=True))
        
        return manifest
    
    def save_results(self, extraction_results: Dict[str, Any], manifest: Dict[str, Any]):
        """Save extraction results and manifest to files."""
        
        # Save detailed extraction results
        results_file = self.output_dir / "extraction_results.json"
        with open(results_file, 'w') as f:
            json.dump(extraction_results, f, indent=2, default=str)
        
        # Save dataset manifest
        manifest_file = self.output_dir / "dataset_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Save failed extractions for review
        if extraction_results['statistics']['extraction_errors']:
            errors_file = self.output_dir / "extraction_errors.json"
            with open(errors_file, 'w') as f:
                json.dump(extraction_results['statistics']['extraction_errors'], f, indent=2)
        
        logger.info(f"💾 Results saved to {self.output_dir}")
        logger.info(f"   📊 Dataset manifest: {manifest_file}")
        logger.info(f"   📝 Extraction results: {results_file}")
        if extraction_results['statistics']['extraction_errors']:
            logger.info(f"   ❌ Extraction errors: {errors_file}")
    
    def print_summary(self, manifest: Dict[str, Any]):
        """Print a comprehensive summary of the extracted dataset."""
        
        print("\n" + "="*80)
        print("🎯 MALICIOUS SAMPLE EXTRACTION SUMMARY")
        print("="*80)
        
        print(f"📅 Extraction Date: {manifest['extraction_date']}")
        print(f"📦 Total Packages: {manifest['total_packages']:,}")
        print(f"❌ Failed Packages: {manifest['failed_packages']:,}")
        
        print(f"\n🏗️  ECOSYSTEMS:")
        for ecosystem, stats in manifest['ecosystems'].items():
            print(f"   {ecosystem.upper()}:")
            print(f"     📦 Packages: {stats['total_packages']:,}")
            print(f"     📄 Files: {stats['total_files']:,}")
            print(f"     💾 Size: {stats['total_size_bytes'] / (1024*1024):.1f} MB")
            for category, count in stats['categories'].items():
                print(f"       {category}: {count:,}")
        
        print(f"\n📊 CATEGORIES:")
        for category, count in manifest['categories'].items():
            print(f"   {category}: {count:,} ({count/manifest['total_packages']*100:.1f}%)")
        
        print(f"\n📄 FILE TYPES (Top 10):")
        for ext, count in list(manifest['file_type_distribution'].items())[:10]:
            ext_name = ext if ext != 'no_extension' else '(no extension)'
            print(f"   {ext_name}: {count:,}")
        
        size_stats = manifest['size_statistics']
        print(f"\n💾 SIZE STATISTICS:")
        print(f"   Total: {size_stats['total_size_bytes'] / (1024*1024):.1f} MB")
        print(f"   Average per package: {size_stats['average_package_size'] / 1024:.1f} KB")
        print(f"   Largest: {size_stats['largest_package']['name']} "
              f"({size_stats['largest_package']['size'] / 1024:.1f} KB)")
        print(f"   Smallest: {size_stats['smallest_package']['name']} "
              f"({size_stats['smallest_package']['size']:.0f} bytes)")
        
        print("\n" + "="*80)
        print("✅ EXTRACTION COMPLETE - Dataset ready for ICN training!")
        print("="*80)


def main():
    """Main extraction pipeline."""
    
    parser = argparse.ArgumentParser(description="Extract malicious package samples for ICN training")
    parser.add_argument("--samples-dir", default="malicious-software-packages-dataset/samples",
                       help="Source samples directory")
    parser.add_argument("--output-dir", default="data/extracted_samples", 
                       help="Output directory for extracted samples")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum number of parallel workers")
    parser.add_argument("--dry-run", action="store_true",
                       help="Discover samples without extracting")
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("🚀 Starting malicious sample extraction pipeline")
    logger.info(f"   Source: {args.samples_dir}")
    logger.info(f"   Output: {args.output_dir}")
    logger.info(f"   Workers: {args.max_workers}")
    
    # Initialize extractor
    extractor = MaliciousSampleExtractor(
        samples_dir=args.samples_dir,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    try:
        # Discover all samples
        samples = extractor.discover_samples()
        
        if args.dry_run:
            logger.info("🔍 Dry run complete - no extraction performed")
            return
        
        # Extract all samples
        extraction_results = extractor.extract_all_samples(samples)
        
        # Generate dataset manifest
        manifest = extractor.generate_dataset_manifest(extraction_results)
        
        # Save results
        extractor.save_results(extraction_results, manifest)
        
        # Print summary
        extractor.print_summary(manifest)
        
        logger.info("🎉 Malicious sample extraction pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⚠️  Extraction interrupted by user")
    except Exception as e:
        logger.error(f"💥 Extraction pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()