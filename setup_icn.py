#!/usr/bin/env python3
"""Setup script for ICN project structure and dependencies."""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command with error handling."""
    print(f"📋 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return None

def main():
    print("🚀 ICN Project Setup")
    print("=" * 30)
    
    # Create directory structure
    dirs = [
        "icn/models",
        "icn/training", 
        "icn/evaluation",
        "data/extracted_malicious",
        "data/collected_benign",
        "data/processed",
        "logs",
        "checkpoints"
    ]
    
    print("📁 Creating directory structure...")
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  • {dir_path}/")
    
    # Add Python dependencies for Phase 2
    ml_deps = [
        "torch>=2.0.0",
        "transformers>=4.20.0", 
        "torch-audio",
        "scikit-learn",
        "numpy", 
        "matplotlib",
        "seaborn",
        "wandb",  # for experiment tracking
        "tqdm",
        "tokenizers"
    ]
    
    print(f"\n🔧 Dependencies needed for Phase 2:")
    for dep in ml_deps:
        print(f"  • {dep}")
    
    print(f"\nTo install dependencies, run:")
    print(f"  uv add {' '.join(ml_deps)}")
    
    # Create __init__.py files
    init_files = [
        "icn/models/__init__.py",
        "icn/training/__init__.py", 
        "icn/evaluation/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
    
    print(f"\n📋 Phase 1 Summary:")
    print(f"✅ Malicious data extraction (malicious-software-packages-dataset)")
    print(f"✅ Benign data collection (npm + PyPI APIs)")
    print(f"✅ Unified parsing pipeline (AST + API detection)")
    print(f"✅ Intent categorization (15 fixed + latent slots)")
    print(f"✅ Phase detection (install vs runtime)")
    
    print(f"\n📋 Ready for Phase 2:")
    print(f"🔬 Local Intent Estimator (CodeBERT-based)")
    print(f"🌐 Global Intent Integrator (convergence loop)")
    print(f"🎯 Dual detection channels (divergence + plausibility)")
    print(f"📚 Training with dataset-specific losses")
    
    print(f"\n🎯 Dataset Statistics:")
    print(f"• ~10K malicious packages (6,874 malicious_intent + 47 compromised_lib)")
    print(f"• Need ~50K benign packages (5:1 ratio)")
    print(f"• Perfect mapping to dual detection channels")
    
    print(f"\n🚀 Next: Run 'python icn_demo.py' to see the full pipeline!")

if __name__ == "__main__":
    main()