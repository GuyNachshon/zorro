#!/usr/bin/env python3
"""
Test PEFT model support in evaluation framework.
"""

import asyncio
from evaluation.config import load_config
from evaluation.runner import EvaluationRunner

async def test_peft_config_loading():
    """Test that PEFT models can be loaded from config."""
    print("🧪 Testing PEFT configuration loading...")

    try:
        # Load the PEFT config
        config = load_config('evaluation/configs/huggingface_peft.yaml')
        print(f"✅ Config loaded: {config.name}")
        print(f"   Models: {len(config.models)}")

        # Find the PEFT model
        peft_models = [m for m in config.models if m.use_peft]
        if peft_models:
            peft_model = peft_models[0]
            print(f"✅ PEFT model found: {peft_model.name}")
            print(f"   Base model: {peft_model.hf_base_model_id}")
            print(f"   Adapter: {peft_model.hf_adapter_id}")
        else:
            print("❌ No PEFT models found in config")
            return False

        # Validate config
        issues = config.validate()
        if issues:
            print(f"❌ Validation issues: {issues}")
            return False
        else:
            print("✅ Configuration validation passed")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_peft_model_creation():
    """Test creating a PEFT model instance (without actually loading it)."""
    print("\n🧪 Testing PEFT model creation...")

    try:
        from icn.evaluation.benchmark_framework import HuggingFaceModel

        # Create a PEFT model instance
        peft_model = HuggingFaceModel(
            model_name="test_peft_model",
            model_id="test_peft_model",
            base_model_id="microsoft/codebert-base-mlm",
            adapter_id="endorlabs/malicious-package-classifier-bert-mal-only",
            use_peft=True
        )

        print(f"✅ PEFT model instance created: {peft_model.model_name}")
        print(f"   Use PEFT: {peft_model.use_peft}")
        print(f"   Base model: {peft_model.base_model_id}")
        print(f"   Adapter: {peft_model.adapter_id}")

        # Test regular HuggingFace model too
        regular_model = HuggingFaceModel(
            model_name="regular_model",
            model_id="microsoft/codebert-base-mlm",
            use_peft=False
        )

        print(f"✅ Regular HF model instance created: {regular_model.model_name}")
        print(f"   Use PEFT: {regular_model.use_peft}")

        return True

    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all PEFT support tests."""
    print("🚀 Testing PEFT Support in Evaluation Framework")
    print("=" * 50)

    success = True

    # Test 1: Config loading
    if not await test_peft_config_loading():
        success = False

    # Test 2: Model creation
    if not test_peft_model_creation():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("🎉 All PEFT support tests passed!")
        print("\nTo use PEFT models in evaluation:")
        print("1. Create a model config with use_peft: true")
        print("2. Specify hf_base_model_id and hf_adapter_id")
        print("3. Make sure 'peft' library is installed: uv add peft")
        print("\nExample PEFT model config:")
        print("```yaml")
        print("- name: endorlabs_classifier")
        print("  type: huggingface")
        print("  use_peft: true")
        print("  hf_base_model_id: microsoft/codebert-base-mlm")
        print("  hf_adapter_id: endorlabs/malicious-package-classifier-bert-mal-only")
        print("```")
    else:
        print("❌ Some tests failed.")

    return success

if __name__ == "__main__":
    import sys
    sys.exit(0 if asyncio.run(main()) else 1)