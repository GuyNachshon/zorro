#!/usr/bin/env python3
"""
Simple test to verify training functions work with minimal complexity.
"""

def test_simple_imports():
    """Test basic imports work."""
    print("🧪 Testing basic imports...")

    try:
        from train_amil import create_sample_data
        print("✅ AMIL sample data import successful")

        from train_cpg import create_sample_data as cpg_sample
        print("✅ CPG sample data import successful")

        from train_neobert import train_neobert_model
        print("✅ NeoBERT import successful")

        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_sample_data_creation():
    """Test sample data creation works."""
    print("\n🧪 Testing sample data creation...")

    try:
        from train_amil import create_sample_data
        train_samples, val_samples = create_sample_data()
        print(f"✅ AMIL sample data: {len(train_samples)} train, {len(val_samples)} val")

        from train_cpg import create_sample_data as cpg_sample
        train_cpg, val_cpg = cpg_sample()
        print(f"✅ CPG sample data: {len(train_cpg)} train, {len(val_cpg)} val")

        return True
    except Exception as e:
        print(f"❌ Sample data creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_neobert_only():
    """Test only NeoBERT since it uses placeholder implementation."""
    print("\n🧪 Testing NeoBERT training (placeholder)...")

    try:
        from train_neobert import train_neobert_model
        result = train_neobert_model(
            save_dir="test_checkpoints/neobert_simple",
            log_level="ERROR",  # Minimal logging
            batch_size=1,
            max_length=64
        )
        print(f"✅ NeoBERT completed: {result.get('training_completed', False)}")
        return True
    except Exception as e:
        print(f"❌ NeoBERT training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simplified tests."""
    print("🚀 Simple Training Function Tests")
    print("=" * 50)

    success = True

    # Test 1: Basic imports
    if not test_simple_imports():
        success = False
        return success

    # Test 2: Sample data creation
    if not test_sample_data_creation():
        success = False
        return success

    # Test 3: NeoBERT (simplest case)
    if not test_neobert_only():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("🎉 Basic functionality works!")
        print("Issues are likely in complex AMIL/CPG training loops.")
        print("Consider:")
        print("• Using CPU-only mode to avoid CUDA issues")
        print("• Simplifying the training pipeline")
        print("• Testing individual components in isolation")
    else:
        print("❌ Basic functionality has issues.")

    return success

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)