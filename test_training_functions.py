#!/usr/bin/env python3
"""
Test script to verify all training functions work without argparse.
"""

def test_imports():
    """Test that all training functions can be imported."""
    print("🧪 Testing training function imports...")

    try:
        from train_amil import train_amil_model
        print("✅ AMIL import successful")
    except Exception as e:
        print(f"❌ AMIL import failed: {e}")
        return False

    try:
        from train_cpg import train_cpg_model
        print("✅ CPG import successful")
    except Exception as e:
        print(f"❌ CPG import failed: {e}")
        return False

    try:
        from train_neobert import train_neobert_model
        print("✅ NeoBERT import successful")
    except Exception as e:
        print(f"❌ NeoBERT import failed: {e}")
        return False

    return True


def test_function_calls():
    """Test that training functions can be called programmatically."""
    print("\n🧪 Testing training function calls...")

    # Import functions
    from train_amil import train_amil_model
    from train_cpg import train_cpg_model
    from train_neobert import train_neobert_model

    # Test AMIL (quick test)
    try:
        print("🎯 Testing AMIL training...")
        results = train_amil_model(
            save_dir="test_checkpoints/amil",
            log_level="WARNING",  # Reduce log noise
            batch_size=4,         # Small batch for quick test
        )
        print(f"✅ AMIL training completed: {results.get('training_completed', False)}")
    except Exception as e:
        print(f"❌ AMIL training failed: {e}")
        return False

    # Test CPG (quick test)
    try:
        print("🎯 Testing CPG training...")
        results = train_cpg_model(
            save_dir="test_checkpoints/cpg",
            log_level="WARNING",
            batch_size=4,
            hidden_dim=64,  # Smaller for quick test
        )
        print(f"✅ CPG training completed: {results.get('training_completed', False)}")
    except Exception as e:
        print(f"❌ CPG training failed: {e}")
        return False

    # Test NeoBERT (quick test)
    try:
        print("🎯 Testing NeoBERT training...")
        results = train_neobert_model(
            save_dir="test_checkpoints/neobert",
            log_level="WARNING",
            batch_size=2,
            max_length=128,  # Smaller for quick test
        )
        print(f"✅ NeoBERT training completed: {results.get('training_completed', False)}")
    except Exception as e:
        print(f"❌ NeoBERT training failed: {e}")
        return False

    return True


def test_evaluation_integration():
    """Test integration with evaluation package."""
    print("\n🧪 Testing evaluation package integration...")

    try:
        from evaluation.config import EvaluationConfig, ModelConfig

        # Create test config with local models
        config = EvaluationConfig(
            name="test_local_models",
            models=[
                ModelConfig(name="test_amil", type="amil", enabled=True),
                ModelConfig(name="test_cpg", type="cpg", enabled=True),
                ModelConfig(name="test_neobert", type="neobert", enabled=True)
            ]
        )

        issues = config.validate()
        if issues:
            print(f"⚠️ Configuration validation warnings: {issues}")
        else:
            print("✅ Configuration validation passed")

        print("✅ Evaluation package integration successful")
        return True

    except Exception as e:
        print(f"❌ Evaluation integration failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Training Functions Without Argparse")
    print("=" * 60)

    success = True

    # Test imports
    if not test_imports():
        success = False

    # Test function calls
    if not test_function_calls():
        success = False

    # Test evaluation integration
    if not test_evaluation_integration():
        success = False

    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! Training functions work without argparse.")
        print("\nYou can now:")
        print("• Call train_amil_model(), train_cpg_model(), train_neobert_model() directly")
        print("• Use them in YAML configurations")
        print("• Let the evaluation package train models automatically")
    else:
        print("❌ Some tests failed. Check the errors above.")

    return success


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)