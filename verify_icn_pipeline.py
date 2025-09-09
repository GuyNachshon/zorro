#!/usr/bin/env python3
"""
ICN Full Pipeline Verification Script
Tests all components with PyTorch dependencies installed
"""

import torch
import torch.nn.functional as F
import sys
import time
from pathlib import Path
import traceback

# Add ICN modules to path
sys.path.append(str(Path(__file__).parent))

from icn.models.icn_model import ICNModel, ICNInput
from icn.models.local_estimator import IntentVocabulary, LocalIntentEstimator
from icn.models.global_integrator import GlobalIntentIntegrator
from icn.models.detection import DualDetectionSystem
from icn.training.losses import ICNLossComputer, SampleType, BenignManifoldModel
from icn.data.malicious_extractor import MaliciousExtractor
from icn.data.benign_collector import BenignCollector
from icn.parsing.unified_parser import UnifiedParser


def test_pytorch_installation():
    """Test PyTorch installation and GPU availability."""
    print("🔍 Testing PyTorch Installation...")
    
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test basic tensor operations
    x = torch.randn(3, 4)
    y = torch.matmul(x, x.T)
    print(f"  Basic tensor ops: ✅ Working (result shape: {y.shape})")
    
    return torch.cuda.is_available()


def test_transformers_integration():
    """Test transformers library integration."""
    print("\n🤖 Testing Transformers Integration...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        print(f"  CodeBERT tokenizer: ✅ Loaded (vocab size: {tokenizer.vocab_size})")
        
        # Test model loading (small test)
        model = AutoModel.from_pretrained("microsoft/codebert-base")
        print(f"  CodeBERT model: ✅ Loaded (parameters: {sum(p.numel() for p in model.parameters()):,})")
        
        # Test tokenization
        code = "def hello_world(): print('Hello, World!')"
        tokens = tokenizer.encode(code, max_length=128, padding='max_length', truncation=True)
        print(f"  Tokenization: ✅ Working (tokens: {len(tokens)})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Transformers integration failed: {e}")
        return False


def test_local_intent_estimator():
    """Test Local Intent Estimator with real PyTorch."""
    print("\n🧠 Testing Local Intent Estimator...")
    
    try:
        # Create model with pretrained CodeBERT
        model = LocalIntentEstimator(
            use_pretrained=True,
            n_fixed_intents=15,
            n_latent_intents=10
        )
        print(f"  Model creation: ✅ Success")
        
        # Test forward pass
        batch_size = 2
        seq_length = 128
        
        input_ids = torch.randint(0, 50000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        phase_ids = torch.randint(0, 3, (batch_size,))
        api_features = torch.randn(batch_size, 15)
        ast_features = torch.randn(batch_size, 50)
        
        start_time = time.time()
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                phase_ids=phase_ids,
                api_features=api_features,
                ast_features=ast_features
            )
        inference_time = time.time() - start_time
        
        print(f"  Forward pass: ✅ Success ({inference_time:.3f}s)")
        print(f"  Fixed intents shape: {output.fixed_intent_dist.shape}")
        print(f"  Latent intents shape: {output.latent_intent_dist.shape}")
        print(f"  Embeddings shape: {output.unit_embeddings.shape}")
        
        # Test intent vocabulary
        vocab = IntentVocabulary()
        test_intents = ["net.outbound", "proc.spawn", "eval"]
        encoded = vocab.encode(test_intents)
        decoded = vocab.decode(encoded)
        print(f"  Intent vocab: ✅ Working ({len(decoded)} intents decoded)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Local Intent Estimator failed: {e}")
        traceback.print_exc()
        return False


def test_global_intent_integrator():
    """Test Global Intent Integrator with convergence loop."""
    print("\n🌐 Testing Global Intent Integrator...")
    
    try:
        model = GlobalIntentIntegrator(
            embedding_dim=768,
            n_fixed_intents=15,
            n_latent_intents=10,
            max_iterations=4
        )
        print(f"  Model creation: ✅ Success")
        
        # Test convergence loop
        batch_size = 2
        max_units = 4
        embedding_dim = 768
        n_total_intents = 25
        
        local_embeddings = torch.randn(batch_size, max_units, embedding_dim)
        local_intent_dists = F.softmax(torch.randn(batch_size, max_units, n_total_intents), dim=-1)
        unit_masks = torch.ones(batch_size, max_units, dtype=torch.bool)
        unit_masks[1, 3] = False  # Variable lengths
        
        start_time = time.time()
        with torch.no_grad():
            output = model(
                local_embeddings=local_embeddings,
                local_intent_dists=local_intent_dists,
                unit_masks=unit_masks,
                return_history=True
            )
        convergence_time = time.time() - start_time
        
        print(f"  Convergence loop: ✅ Success ({convergence_time:.3f}s)")
        print(f"  Converged: {output.converged} in {output.final_iteration} iterations")
        print(f"  History length: {len(output.convergence_history)}")
        
        # Test metrics computation
        conv_metrics = model.get_convergence_metrics(output.convergence_history)
        div_metrics = model.compute_divergence_metrics(
            local_intent_dists, output.final_global_intent, unit_masks
        )
        print(f"  Metrics computation: ✅ Success")
        print(f"    Mean divergence: {div_metrics['mean_divergence'].mean().item():.4f}")
        print(f"    Iterations to converge: {conv_metrics['iterations_to_converge']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Global Intent Integrator failed: {e}")
        traceback.print_exc()
        return False


def test_dual_detection_system():
    """Test Dual Detection System."""
    print("\n🕵️ Testing Dual Detection System...")
    
    try:
        detector = DualDetectionSystem(embedding_dim=768)
        print(f"  Model creation: ✅ Success")
        
        # Prepare test data
        batch_size = 3
        max_units = 4
        embedding_dim = 768
        n_intents = 25
        
        local_embeddings = torch.randn(batch_size, max_units, embedding_dim)
        global_embeddings = torch.randn(batch_size, embedding_dim)
        global_intent_dists = F.softmax(torch.randn(batch_size, n_intents), dim=-1)
        unit_masks = torch.ones(batch_size, max_units, dtype=torch.bool)
        attention_weights = F.softmax(torch.randn(batch_size, max_units), dim=-1)
        
        # Mock convergence and divergence metrics
        convergence_metrics = {
            'iterations_to_converge': 3,
            'final_drift': 0.02,
            'converged': True
        }
        
        divergence_metrics = {
            'mean_divergence': torch.rand(batch_size) * 0.5,
            'max_divergence': torch.rand(batch_size) * 0.5 + 0.5,
            'per_unit_divergence': torch.rand(batch_size, max_units)
        }
        
        # Fit benign manifold
        benign_embeddings = torch.randn(50, embedding_dim)
        detector.plausibility_detector.fit_benign_manifold(benign_embeddings)
        print(f"  Benign manifold fitting: ✅ Success")
        
        # Test detection
        start_time = time.time()
        with torch.no_grad():
            output = detector(
                convergence_metrics=convergence_metrics,
                divergence_metrics=divergence_metrics,
                local_embeddings=local_embeddings,
                unit_masks=unit_masks,
                attention_weights=attention_weights,
                global_embeddings=global_embeddings,
                global_intent_dists=global_intent_dists,
                phase_violations=torch.randint(0, 3, (batch_size,)),
                latent_activations=torch.rand(batch_size, 10)
            )
        detection_time = time.time() - start_time
        
        print(f"  Detection pipeline: ✅ Success ({detection_time:.3f}s)")
        print(f"  Final scores: {output.final_scores.tolist()}")
        print(f"  Predictions: {output.final_predictions.tolist()}")
        
        # Test explanations
        for i, explanation in enumerate(output.explanations[:1]):  # Show first explanation
            print(f"  Sample {i} explanation:")
            print(f"    Primary channel: {explanation['primary_channel']}")
            print(f"    Final score: {explanation['final_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Dual Detection System failed: {e}")
        traceback.print_exc()
        return False


def test_complete_icn_model():
    """Test the complete ICN model end-to-end."""
    print("\n🚀 Testing Complete ICN Model...")
    
    try:
        # Create model (using smaller dims for faster testing)
        model = ICNModel(
            embedding_dim=256,  # Smaller for testing
            hidden_dim=128,
            use_pretrained=False,  # Custom transformer for speed
            max_iterations=3
        )
        print(f"  Model creation: ✅ Success")
        
        # Prepare batch input
        batch_size = 2
        max_seq_len = 64
        
        input_ids_list = []
        attention_masks_list = []
        phase_ids_list = []
        api_features_list = []
        ast_features_list = []
        
        for i in range(batch_size):
            n_units = 2 + i  # Variable units per package
            
            input_ids_list.append(torch.randint(0, 1000, (n_units, max_seq_len)))
            attention_masks_list.append(torch.ones(n_units, max_seq_len))
            phase_ids_list.append(torch.randint(0, 3, (n_units,)))
            api_features_list.append(torch.randn(n_units, 15))
            ast_features_list.append(torch.randn(n_units, 50))
        
        batch_input = ICNInput(
            input_ids_list=input_ids_list,
            attention_masks_list=attention_masks_list,
            phase_ids_list=phase_ids_list,
            api_features_list=api_features_list,
            ast_features_list=ast_features_list,
            manifest_embeddings=torch.randn(batch_size, 256),
            sample_types=['benign', 'malicious_intent'],
            malicious_labels=torch.tensor([0, 1], dtype=torch.float)
        )
        
        # Fit benign manifold
        benign_embeddings = torch.randn(50, 256)
        model.fit_benign_manifold(benign_embeddings)
        print(f"  Benign manifold fitting: ✅ Success")
        
        # End-to-end inference
        start_time = time.time()
        with torch.no_grad():
            output = model(batch_input)
        e2e_time = time.time() - start_time
        
        print(f"  End-to-end inference: ✅ Success ({e2e_time:.3f}s)")
        print(f"  Malicious scores: {output.malicious_scores.tolist()}")
        print(f"  Predictions: {output.malicious_predictions.tolist()}")
        print(f"  Converged: {output.global_output.converged}")
        
        # Test interpretations
        interpretations = model.interpret_predictions(output)
        print(f"  Interpretations: ✅ Generated ({len(interpretations)} samples)")
        
        for i, interp in enumerate(interpretations):
            print(f"    Package {i}: {interp['prediction']} ({interp['confidence']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Complete ICN Model failed: {e}")
        traceback.print_exc()
        return False


def test_training_losses():
    """Test training loss computation."""
    print("\n📊 Testing Training Losses...")
    
    try:
        loss_computer = ICNLossComputer()
        manifold_model = BenignManifoldModel(256)
        
        # Create mock data
        batch_size = 3
        embedding_dim = 256
        
        # Fit benign manifold
        benign_embeddings = torch.randn(100, embedding_dim)
        manifold_model.fit(benign_embeddings)
        print(f"  Manifold fitting: ✅ Success")
        
        # Create mock local outputs
        from icn.models.local_estimator import LocalIntentOutput
        local_outputs = []
        
        for i in range(batch_size):
            units = []
            for j in range(2):  # 2 units per package
                unit_output = LocalIntentOutput(
                    fixed_intent_dist=F.softmax(torch.randn(15), dim=-1),
                    latent_intent_dist=F.softmax(torch.randn(10), dim=-1),
                    unit_embeddings=torch.randn(embedding_dim)
                )
                units.append(unit_output)
            local_outputs.append(units)
        
        # Create other inputs
        global_output = F.softmax(torch.randn(batch_size, 25), dim=-1)
        global_embeddings = torch.randn(batch_size, embedding_dim)
        sample_types = [SampleType.BENIGN, SampleType.COMPROMISED_LIB, SampleType.MALICIOUS_INTENT]
        malicious_labels = torch.tensor([0, 1, 1], dtype=torch.float)
        
        # Compute losses
        losses = loss_computer.compute_losses(
            local_outputs=local_outputs,
            global_output=global_output,
            global_embeddings=global_embeddings,
            sample_types=sample_types,
            malicious_labels=malicious_labels,
            benign_manifold=manifold_model.get_prototypes(torch.device('cpu'))
        )
        
        print(f"  Loss computation: ✅ Success")
        print(f"  Total loss: {losses['total'].item():.4f}")
        
        # Show individual losses
        for loss_name, loss_value in losses.items():
            if loss_value is not None and 'total' not in loss_name:
                print(f"    {loss_name}: {loss_value.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Training losses failed: {e}")
        traceback.print_exc()
        return False


def test_data_pipeline():
    """Test data extraction and parsing pipeline."""
    print("\n📁 Testing Data Pipeline...")
    
    try:
        # Test malicious extractor
        dataset_path = Path("/Users/guynachshon/Documents/baddon-ai/zorro/malicious-software-packages-dataset")
        if dataset_path.exists():
            extractor = MaliciousExtractor(str(dataset_path))
            manifests = extractor.load_manifests()
            print(f"  Malicious extractor: ✅ Working ({len(manifests)} ecosystems)")
            
            categories = extractor.categorize_packages(manifests)
            print(f"    Malicious intent: {len(categories['malicious_intent'])}")
            print(f"    Compromised lib: {len(categories['compromised_lib'])}")
        else:
            print(f"  Malicious extractor: ⚠️  Dataset not found at {dataset_path}")
        
        # Test benign collector (just API connectivity)
        collector = BenignCollector()
        npm_popular = collector.npm.get_popular_packages(3)
        print(f"  Benign collector: ✅ Working (found {len(npm_popular)} npm packages)")
        
        # Test unified parser
        parser = UnifiedParser()
        test_code = 'import os\ndef test(): os.system("ls")'
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            temp_path = Path(f.name)
        
        try:
            units = parser._parse_python_file(temp_path, test_code, "runtime")
            print(f"  Unified parser: ✅ Working ({len(units)} units extracted)")
            
            for unit in units[:1]:  # Show first unit
                print(f"    API calls: {unit.api_calls}")
                print(f"    Categories: {unit.api_categories}")
                
        finally:
            temp_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data pipeline failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run complete pipeline verification."""
    print("🚀 ICN Full Pipeline Verification")
    print("=" * 60)
    
    # Track results
    results = {}
    
    # Test each component
    tests = [
        ("PyTorch Installation", test_pytorch_installation),
        ("Transformers Integration", test_transformers_integration),
        ("Local Intent Estimator", test_local_intent_estimator),
        ("Global Intent Integrator", test_global_intent_integrator),
        ("Dual Detection System", test_dual_detection_system),
        ("Complete ICN Model", test_complete_icn_model),
        ("Training Losses", test_training_losses),
        ("Data Pipeline", test_data_pipeline)
    ]
    
    total_start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  ❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    total_time = time.time() - total_start_time
    
    # Summary
    print(f"\n📊 Verification Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:<25} {status}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    
    if passed == total:
        print(f"\n🎉 ICN Pipeline: FULLY OPERATIONAL!")
        print(f"   Ready for Phase 3B: Dataset Preparation and Training")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Extract all malicious samples: python extract_malicious_samples.py")
        print(f"   2. Collect benign packages: python collect_benign_samples.py")
        print(f"   3. Start curriculum training: python train_icn.py")
        
    else:
        print(f"\n⚠️  Pipeline Issues Detected")
        print(f"   Please fix failing tests before proceeding to training")
        
        failed_tests = [name for name, result in results.items() if not result]
        print(f"   Failed tests: {', '.join(failed_tests)}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)