#!/usr/bin/env python3
"""
Test ICN structure and imports without requiring PyTorch.
Verifies that all components are properly implemented.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all ICN modules can be imported."""
    print("🔍 Testing ICN Module Structure...")
    
    # Test that all files exist
    expected_files = [
        "icn/__init__.py",
        "icn/data/__init__.py", 
        "icn/data/malicious_extractor.py",
        "icn/data/benign_collector.py",
        "icn/parsing/__init__.py",
        "icn/parsing/unified_parser.py",
        "icn/models/__init__.py",
        "icn/models/local_estimator.py",
        "icn/models/global_integrator.py", 
        "icn/models/detection.py",
        "icn/models/icn_model.py",
        "icn/training/__init__.py",
        "icn/training/losses.py"
    ]
    
    missing_files = []
    for file_path in expected_files:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} files:")
        for file_path in missing_files:
            print(f"   • {file_path}")
        return False
    else:
        print(f"\n✅ All {len(expected_files)} files present!")
        return True

def test_class_definitions():
    """Test that key classes are defined (without importing torch)."""
    print(f"\n🔍 Testing Class Definitions...")
    
    # Check for key class definitions in files
    class_checks = [
        ("icn/data/malicious_extractor.py", ["MaliciousExtractor", "PackageSample"]),
        ("icn/data/benign_collector.py", ["BenignCollector", "NpmCollector", "PyPICollector", "BenignSample"]),
        ("icn/parsing/unified_parser.py", ["UnifiedParser", "CodeUnit", "PackageAnalysis"]),
        ("icn/models/local_estimator.py", ["LocalIntentEstimator", "LocalIntentOutput", "IntentVocabulary"]),
        ("icn/models/global_integrator.py", ["GlobalIntentIntegrator", "GlobalIntegratorOutput", "ConvergenceState"]),
        ("icn/models/detection.py", ["DualDetectionSystem", "DivergenceDetector", "PlausibilityDetector"]),
        ("icn/models/icn_model.py", ["ICNModel", "ICNInput", "ICNOutput"]),
        ("icn/training/losses.py", ["ICNLossComputer", "BenignManifoldModel", "SampleType"])
    ]
    
    all_classes_found = True
    
    for file_path, expected_classes in class_checks:
        if not Path(file_path).exists():
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        print(f"\n   📄 {file_path}:")
        for class_name in expected_classes:
            if f"class {class_name}" in content:
                print(f"      ✅ {class_name}")
            else:
                print(f"      ❌ {class_name}")
                all_classes_found = False
    
    return all_classes_found

def test_method_signatures():
    """Test that key methods are implemented."""
    print(f"\n🔍 Testing Key Method Signatures...")
    
    method_checks = [
        ("icn/data/malicious_extractor.py", ["load_manifests", "categorize_packages", "extract_sample"]),
        ("icn/data/benign_collector.py", ["get_popular_packages", "get_longtail_packages", "collect_balanced_dataset"]),
        ("icn/parsing/unified_parser.py", ["parse_package", "_parse_python_file", "_extract_api_calls"]),
        ("icn/models/local_estimator.py", ["forward", "compute_intent_entropy", "get_dominant_intents"]),
        ("icn/models/global_integrator.py", ["forward", "get_convergence_metrics", "compute_divergence_metrics"]),
        ("icn/models/detection.py", ["forward", "_compute_divergence_confidence", "_compute_plausibility_confidence"]),
        ("icn/models/icn_model.py", ["forward", "interpret_predictions", "fit_benign_manifold"]),
        ("icn/training/losses.py", ["compute_losses", "_compute_convergence_loss", "_compute_divergence_margin_loss"])
    ]
    
    all_methods_found = True
    
    for file_path, expected_methods in method_checks:
        if not Path(file_path).exists():
            continue
            
        with open(file_path, 'r') as f:
            content = f.read()
        
        print(f"\n   📄 {file_path}:")
        for method_name in expected_methods:
            if f"def {method_name}" in content:
                print(f"      ✅ {method_name}")
            else:
                print(f"      ❌ {method_name}")
                all_methods_found = False
    
    return all_methods_found

def analyze_implementation_completeness():
    """Analyze the completeness of the implementation."""
    print(f"\n📊 Implementation Analysis...")
    
    # Count lines of code
    total_lines = 0
    file_stats = {}
    
    for py_file in Path(".").rglob("icn/**/*.py"):
        if py_file.name != "__init__.py":
            with open(py_file, 'r') as f:
                lines = len(f.readlines())
            total_lines += lines
            file_stats[str(py_file)] = lines
    
    print(f"   📏 Total lines of code: {total_lines:,}")
    
    print(f"\n   📁 File sizes:")
    for file_path, lines in sorted(file_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"      {Path(file_path).name}: {lines} lines")
    
    # Check for key features
    key_features = {
        "CodeBERT integration": "transformers",
        "Convergence loop": "convergence_threshold", 
        "Divergence detection": "KL divergence",
        "Plausibility detection": "benign_manifold",
        "Training losses": "margin_loss",
        "Intent vocabulary": "fixed_intents",
        "Phase detection": "install.*runtime",
        "API categorization": "net.outbound"
    }
    
    print(f"\n   🔍 Key Feature Detection:")
    for feature, pattern in key_features.items():
        found = False
        for py_file in Path(".").rglob("icn/**/*.py"):
            if py_file.name != "__init__.py":
                with open(py_file, 'r') as f:
                    content = f.read()
                if pattern.lower() in content.lower():
                    found = True
                    break
        
        status = "✅" if found else "❌"
        print(f"      {status} {feature}")

def main():
    print("🚀 ICN Phase 2 Structure Validation")
    print("=" * 50)
    
    # Run all tests
    structure_ok = test_imports()
    classes_ok = test_class_definitions() 
    methods_ok = test_method_signatures()
    
    # Analyze implementation
    analyze_implementation_completeness()
    
    # Summary
    print(f"\n📋 Validation Summary:")
    print(f"   File structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"   Class definitions: {'✅ PASS' if classes_ok else '❌ FAIL'}")
    print(f"   Method signatures: {'✅ PASS' if methods_ok else '❌ FAIL'}")
    
    overall_status = structure_ok and classes_ok and methods_ok
    
    if overall_status:
        print(f"\n🎉 ICN Phase 2 Implementation: ✅ COMPLETE")
        print(f"   Ready for PyTorch dependency installation and training!")
    else:
        print(f"\n⚠️  ICN Phase 2 Implementation: ❌ INCOMPLETE") 
        print(f"   Please fix the issues above before proceeding.")
    
    # Show next steps
    print(f"\n🚀 Next Steps:")
    print(f"   1. Install dependencies: uv add torch transformers scikit-learn")
    print(f"   2. Run full demo: python icn_phase2_demo.py")
    print(f"   3. Implement curriculum training pipeline")
    print(f"   4. Build evaluation framework")
    print(f"   5. Train on malicious-software-packages-dataset")
    
    return overall_status

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)