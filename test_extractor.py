#!/usr/bin/env python3
"""Quick test of the malicious extractor functionality."""

from icn.data.malicious_extractor import MaliciousExtractor

dataset_path = "/Users/guynachshon/Documents/baddon-ai/zorro/malicious-software-packages-dataset"
extractor = MaliciousExtractor(dataset_path)

# Load manifests only
print("Loading manifests...")
manifests = extractor.load_manifests()

print(f"NPM manifest entries: {len(manifests.get('npm', {}))}")
print(f"PyPI manifest entries: {len(manifests.get('pypi', {}))}")

# Check first few entries
print("\nFirst 5 npm entries:")
npm_items = list(manifests.get('npm', {}).items())[:5]
for name, versions in npm_items:
    category = "malicious_intent" if versions is None else "compromised_lib" 
    print(f"  {name}: {category} ({versions})")

print("\nFirst 5 PyPI entries:")
pypi_items = list(manifests.get('pypi', {}).items())[:5]
for name, versions in pypi_items:
    category = "malicious_intent" if versions is None else "compromised_lib"
    print(f"  {name}: {category} ({versions})")