#!/usr/bin/env python3
"""Count categories in the malicious dataset."""

import json

# Count npm categories
with open("/Users/guynachshon/Documents/baddon-ai/zorro/malicious-software-packages-dataset/samples/npm/manifest.json") as f:
    npm_manifest = json.load(f)

npm_malicious_intent = sum(1 for v in npm_manifest.values() if v is None)
npm_compromised_lib = sum(1 for v in npm_manifest.values() if v is not None)

print(f"NPM:")
print(f"  Malicious intent: {npm_malicious_intent}")
print(f"  Compromised lib: {npm_compromised_lib}")
print(f"  Total: {len(npm_manifest)}")

# Count PyPI categories  
with open("/Users/guynachshon/Documents/baddon-ai/zorro/malicious-software-packages-dataset/samples/pypi/manifest.json") as f:
    pypi_manifest = json.load(f)

pypi_malicious_intent = sum(1 for v in pypi_manifest.values() if v is None)  
pypi_compromised_lib = sum(1 for v in pypi_manifest.values() if v is not None)

print(f"\nPyPI:")
print(f"  Malicious intent: {pypi_malicious_intent}")
print(f"  Compromised lib: {pypi_compromised_lib}")
print(f"  Total: {len(pypi_manifest)}")

print(f"\nOverall:")
print(f"  Total malicious intent: {npm_malicious_intent + pypi_malicious_intent}")
print(f"  Total compromised lib: {npm_compromised_lib + pypi_compromised_lib}")
print(f"  Grand total: {len(npm_manifest) + len(pypi_manifest)}")

# Show some compromised lib examples
print(f"\nCompromised lib examples:")
compromised_examples = [(k, v) for k, v in npm_manifest.items() if v is not None][:3]
for name, versions in compromised_examples:
    print(f"  npm/{name}: versions {versions}")
    
compromised_examples = [(k, v) for k, v in pypi_manifest.items() if v is not None][:3]  
for name, versions in compromised_examples:
    print(f"  pypi/{name}: versions {versions}")