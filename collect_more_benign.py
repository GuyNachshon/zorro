#!/usr/bin/env python3
"""
Quick script to collect more benign packages for training.
Uses the existing BenignCollector to download popular packages.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from icn.data.benign_collector import BenignCollector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Collect benign packages."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect benign packages for training")
    parser.add_argument("--target", type=int, default=10000, help="Target number of packages to collect")
    parser.add_argument("--pypi-popular", type=int, default=3000, help="PyPI popular packages")
    parser.add_argument("--pypi-longtail", type=int, default=3000, help="PyPI longtail packages")
    parser.add_argument("--npm-popular", type=int, default=4000, help="npm popular packages")
    args = parser.parse_args()

    collector = BenignCollector(
        cache_dir="data/benign_samples",
        download_timeout=30
    )

    logger.info(f"🎯 Target: {args.target} total benign packages")
    logger.info("=" * 60)

    # Collect PyPI packages
    logger.info(f"\n📦 Collecting {args.pypi_popular} PyPI popular packages...")
    collector.collect_pypi_popular(max_packages=args.pypi_popular)

    logger.info(f"\n📦 Collecting {args.pypi_longtail} PyPI longtail packages...")
    collector.collect_pypi_longtail(max_packages=args.pypi_longtail)

    # Collect npm packages
    logger.info(f"\n📦 Collecting {args.npm_popular} npm popular packages...")
    collector.collect_npm_popular(max_packages=args.npm_popular)

    logger.info("\n" + "=" * 60)
    logger.info("✅ Benign collection complete!")
    logger.info(f"Collected ~{args.pypi_popular + args.pypi_longtail + args.npm_popular} packages")
    logger.info(f"\nRun training with: --max-benign {args.target}")

if __name__ == "__main__":
    main()