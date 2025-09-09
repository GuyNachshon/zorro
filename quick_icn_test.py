#!/usr/bin/env python3
"""
Quick ICN Pipeline Test - No model downloads
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

def test_basic_components():
    """Test basic ICN components without downloading models."""
    print("🚀 Quick ICN Pipeline Test")
    print("=" * 40)
    
    # Test 1: PyTorch basics
    print("1. PyTorch Installation:")
    print(f"   Version: {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")
    print("   ✅ Working")
    
    # Test 2: ICN imports
    print("\n2. ICN Module Imports:")
    try:
        from icn.models.local_estimator import LocalIntentEstimator, IntentVocabulary
        from icn.models.global_integrator import GlobalIntentIntegrator
        from icn.models.detection import DualDetectionSystem
        from icn.training.losses import ICNLossComputer, SampleType
        from icn.models.icn_model import ICNModel, ICNInput
        print("   ✅ All modules imported successfully")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False
    
    # Test 3: Local Intent Estimator (custom transformer only)
    print("\n3. Local Intent Estimator:")
    try:
        model = LocalIntentEstimator(
            use_pretrained=False,  # No download
            vocab_size=1000,
            embedding_dim=256,
            hidden_dim=128,
            n_layers=2,
            n_heads=8  # 256 / 8 = 32 (divisible)
        )
        
        # Test forward pass
        input_ids = torch.randint(0, 1000, (2, 64))
        attention_mask = torch.ones(2, 64)
        
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        
        print(f"   Fixed intents: {output.fixed_intent_dist.shape}")
        print(f"   Latent intents: {output.latent_intent_dist.shape}")
        print(f"   Embeddings: {output.unit_embeddings.shape}")
        print("   ✅ Working")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Test 4: Global Intent Integrator
    print("\n4. Global Intent Integrator:")
    try:
        integrator = GlobalIntentIntegrator(
            embedding_dim=256,
            hidden_dim=128,
            max_iterations=3
        )
        
        # Test data
        batch_size, max_units, n_intents = 2, 3, 25
        local_embeddings = torch.randn(batch_size, max_units, 256)
        local_intents = F.softmax(torch.randn(batch_size, max_units, n_intents), dim=-1)
        unit_masks = torch.ones(batch_size, max_units, dtype=torch.bool)
        
        with torch.no_grad():
            output = integrator(
                local_embeddings=local_embeddings,
                local_intent_dists=local_intents,
                unit_masks=unit_masks
            )
        
        print(f"   Converged: {output.converged}")
        print(f"   Iterations: {output.final_iteration}")
        print(f"   Global intent: {output.final_global_intent.shape}")
        print("   ✅ Working")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Test 5: Detection System
    print("\n5. Dual Detection System:")
    try:
        detector = DualDetectionSystem(embedding_dim=256)
        
        # Fit benign manifold
        benign_embeddings = torch.randn(50, 256)
        detector.plausibility_detector.fit_benign_manifold(benign_embeddings)
        
        # Test detection (mock data)
        conv_metrics = {'iterations_to_converge': 3, 'final_drift': 0.02, 'converged': True}
        div_metrics = {
            'mean_divergence': torch.rand(2).float(),
            'max_divergence': (torch.rand(2) + 0.5).float(),
            'per_unit_divergence': torch.rand(2, 3).float()
        }
        
        with torch.no_grad():
            output = detector(
                convergence_metrics=conv_metrics,
                divergence_metrics=div_metrics,
                local_embeddings=torch.randn(2, 3, 256),
                unit_masks=torch.ones(2, 3, dtype=torch.bool),
                attention_weights=F.softmax(torch.randn(2, 3), dim=-1),
                global_embeddings=torch.randn(2, 256),
                global_intent_dists=F.softmax(torch.randn(2, 25), dim=-1)
            )
        
        print(f"   Final scores: {output.final_scores.tolist()}")
        print(f"   Predictions: {output.final_predictions.tolist()}")
        print("   ✅ Working")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Test 6: Training Losses
    print("\n6. Training Losses:")
    try:
        loss_computer = ICNLossComputer()
        
        # Mock data
        from icn.models.local_estimator import LocalIntentOutput
        local_outputs = [[
            LocalIntentOutput(
                fixed_intent_dist=F.softmax(torch.randn(15), dim=-1),
                latent_intent_dist=F.softmax(torch.randn(10), dim=-1),
                unit_embeddings=torch.randn(256)
            ) for _ in range(2)
        ] for _ in range(2)]
        
        losses = loss_computer.compute_losses(
            local_outputs=local_outputs,
            global_output=F.softmax(torch.randn(2, 25), dim=-1),
            global_embeddings=torch.randn(2, 256),
            sample_types=[SampleType.BENIGN, SampleType.MALICIOUS_INTENT],
            malicious_labels=torch.tensor([0.0, 1.0])
        )
        
        print(f"   Total loss: {losses['total'].item():.4f}")
        print(f"   Loss components: {len([k for k, v in losses.items() if v is not None])}")
        print("   ✅ Working")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    print("\n🎉 All Tests Passed!")
    print("✅ ICN Pipeline Ready for Training")
    return True

if __name__ == "__main__":
    success = test_basic_components()
    if success:
        print("\n🚀 Next: Set up W&B and build training pipeline")
    sys.exit(0 if success else 1)