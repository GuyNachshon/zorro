#!/usr/bin/env python3
"""
Debug ICN model and data compatibility
Quick test to identify tensor size mismatches.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_model_creation():
    """Test if ICN model can be created successfully."""
    print("🔧 Testing ICN model creation...")
    
    try:
        from icn.models.icn_model import ICNModel
        
        # Create model with standard parameters
        model = ICNModel(
            vocab_size=50265,
            embedding_dim=768,
            n_fixed_intents=15,
            n_latent_intents=10,
            hidden_dim=512,
            max_seq_length=512,
            max_iterations=6,
            convergence_threshold=0.01,
            use_pretrained=True,
            model_name="microsoft/codebert-base"
        )
        
        print(f"✅ Model created successfully")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test parameter access
        print("\n🔍 Testing parameter structure:")
        if hasattr(model, 'local_estimator'):
            print("   ✓ Has local_estimator")
            if hasattr(model.local_estimator, 'encoder'):
                print("   ✓ Has encoder")
                encoder_params = list(model.local_estimator.encoder.parameters())
                print(f"   Encoder parameters: {len(encoder_params)}")
                
                # Check parameter shapes
                for i, param in enumerate(encoder_params[:3]):  # First 3 params
                    print(f"     Param {i}: {param.shape}")
                    
        return model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_data_loading():
    """Test if we can create valid data tensors."""
    print("\n🔧 Testing data tensor creation...")
    
    try:
        # Create sample tensors that match our data
        batch_size = 2
        seq_len = 512
        vocab_size = 50265
        
        # Create input tensors
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len))
        phase_ids = torch.randint(0, 3, (batch_size,))
        api_features = torch.randn((batch_size, 15))
        ast_features = torch.randn((batch_size, 50))
        
        print(f"✅ Created sample tensors:")
        print(f"   input_ids: {input_ids.shape}")
        print(f"   attention_mask: {attention_mask.shape}")
        print(f"   phase_ids: {phase_ids.shape}")
        print(f"   api_features: {api_features.shape}")
        print(f"   ast_features: {ast_features.shape}")
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'phase_ids': phase_ids,
            'api_features': api_features,
            'ast_features': ast_features
        }
        
    except Exception as e:
        print(f"❌ Data tensor creation failed: {e}")
        return None


def test_model_forward():
    """Test model forward pass with sample data."""
    print("\n🔧 Testing model forward pass...")
    
    model = test_model_creation()
    data = test_data_loading()
    
    if model is None or data is None:
        print("❌ Cannot test forward pass - model or data creation failed")
        return False
    
    try:
        from icn.models.icn_model import ICNInput
        
        # Create ICN input
        batch_input = ICNInput(
            input_ids_list=[data['input_ids'][0:1], data['input_ids'][1:2]],  # List of tensors
            attention_masks_list=[data['attention_mask'][0:1], data['attention_mask'][1:2]],
            phase_ids_list=[data['phase_ids'][0:1], data['phase_ids'][1:2]],
            api_features_list=[data['api_features'][0:1], data['api_features'][1:2]],
            ast_features_list=[data['ast_features'][0:1], data['ast_features'][1:2]],
            malicious_labels=torch.tensor([0.0, 1.0]),
            sample_types=["benign", "malicious_intent"]
        )
        
        print("✅ Created ICN input batch")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(batch_input)
            print(f"✅ Forward pass successful!")
            print(f"   Output shape: {output.malicious_scores.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_optimizer_creation():
    """Test optimizer creation with parameter grouping."""
    print("\n🔧 Testing optimizer creation...")
    
    model = test_model_creation()
    if model is None:
        return False
    
    try:
        # Test parameter grouping logic
        encoder_params = []
        if hasattr(model.local_estimator, 'encoder'):
            encoder_params = list(model.local_estimator.encoder.parameters())
        
        print(f"   Encoder params: {len(encoder_params)}")
        
        # Use ID-based comparison (our fix)
        encoder_param_ids = {id(p) for p in encoder_params}
        other_params = [
            p for p in model.parameters() 
            if id(p) not in encoder_param_ids
        ]
        
        print(f"   Other params: {len(other_params)}")
        print(f"   Total params: {len(list(model.parameters()))}")
        
        # Create optimizer
        import torch.optim as optim
        param_groups = []
        
        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': 1e-5,
                'name': 'encoder'
            })
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': 2e-5,
                'name': 'other'
            })
        
        optimizer = optim.AdamW(param_groups, lr=2e-5, weight_decay=0.01)
        print(f"✅ Optimizer created successfully with {len(param_groups)} parameter groups")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimizer creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 ICN Debug Test Suite")
    print("=" * 50)
    
    results = {
        'model_creation': test_model_creation() is not None,
        'data_loading': test_data_loading() is not None,
        'forward_pass': test_model_forward(),
        'optimizer_creation': test_optimizer_creation()
    }
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 ICN model is ready for training!")
    else:
        print("\n🔧 Issues found - fix before training")
    
    return all_passed


if __name__ == "__main__":
    main()