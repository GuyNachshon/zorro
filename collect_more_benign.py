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

    collector = BenignCollector(
        cache_dir="data/benign_samples",
        download_timeout=30
    )

    # Collect PyPI packages
    logger.info("Collecting PyPI popular packages...")
    collector.collect_pypi_popular(max_packages=2000)

    logger.info("Collecting PyPI longtail packages...")
    collector.collect_pypi_longtail(max_packages=2000)

    # Collect npm packages
    logger.info("Collecting npm popular packages...")
    collector.collect_npm_popular(max_packages=2000)

    logger.info("✅ Benign collection complete!")
    logger.info("Run the training again with --max-benign 4000")

if __name__ == "__main__":
    main()