"""
Complete ICN (Intent Convergence Networks) Model.
Integrates all components: Local Estimator, Global Integrator, and Dual Detection.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .local_estimator import LocalIntentEstimator, LocalIntentOutput, IntentVocabulary
from .global_integrator import GlobalIntentIntegrator, GlobalIntegratorOutput
from .detection import DualDetectionSystem, DualDetectionOutput


@dataclass
class ICNInput:
    """Input batch for ICN model."""
    # Per-unit inputs (list of tensors for each package)
    input_ids_list: List[torch.Tensor]  # Tokenized code per unit
    attention_masks_list: List[torch.Tensor]  # Attention masks per unit
    phase_ids_list: List[torch.Tensor]  # Phase indicators per unit
    api_features_list: List[torch.Tensor]  # API category features per unit
    ast_features_list: List[torch.Tensor]  # AST features per unit
    
    # Package-level inputs
    manifest_embeddings: Optional[torch.Tensor] = None  # Manifest embeddings
    sample_types: Optional[List[str]] = None  # For training: benign, compromised_lib, malicious_intent
    
    # Training labels (optional)
    intent_labels: Optional[torch.Tensor] = None  # Weak supervision labels
    malicious_labels: Optional[torch.Tensor] = None  # Binary malicious/benign


@dataclass 
class ICNOutput:
    """Output from complete ICN model."""
    # Final predictions
    malicious_scores: torch.Tensor  # [batch_size] maliciousness scores
    malicious_predictions: torch.Tensor  # [batch_size] binary predictions
    
    # Intermediate outputs
    local_outputs: List[List[LocalIntentOutput]]  # Local outputs per unit
    global_output: GlobalIntegratorOutput  # Global integration results
    detection_output: DualDetectionOutput  # Detection channel results
    
    # For analysis
    convergence_metrics: Dict[str, Any]
    divergence_metrics: Dict[str, torch.Tensor]
    explanations: List[Dict]  # Per-sample explanations


class ICNModel(nn.Module):
    """
    Complete Intent Convergence Networks model for malicious package detection.
    """
    
    def __init__(
        self,
        vocab_size: int = 50265,
        embedding_dim: int = 768,
        n_fixed_intents: int = 15,
        n_latent_intents: int = 10,
        hidden_dim: int = 512,
        max_seq_length: int = 512,
        max_iterations: int = 6,
        convergence_threshold: float = 0.01,
        use_pretrained: bool = True,
        model_name: str = "microsoft/codebert-base"
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_fixed_intents = n_fixed_intents
        self.n_latent_intents = n_latent_intents
        self.n_total_intents = n_fixed_intents + n_latent_intents
        
        # Initialize components
        self.local_estimator = LocalIntentEstimator(
            vocab_size=vocab_size,
            n_fixed_intents=n_fixed_intents,
            n_latent_intents=n_latent_intents,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            max_seq_length=max_seq_length,
            use_pretrained=use_pretrained,
            model_name=model_name
        )
        
        self.global_integrator = GlobalIntentIntegrator(
            embedding_dim=embedding_dim,
            n_fixed_intents=n_fixed_intents,
            n_latent_intents=n_latent_intents,
            hidden_dim=hidden_dim,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold
        )
        
        self.detection_system = DualDetectionSystem(
            embedding_dim=embedding_dim
        )
        
        # Intent vocabulary for interpretation
        self.intent_vocab = IntentVocabulary()
        
        # Training mode flag
        self.training_mode = False
    
    def forward(self, batch_input: ICNInput) -> ICNOutput:
        """
        Forward pass of the complete ICN model.
        
        Args:
            batch_input: ICNInput containing all necessary inputs
            
        Returns:
            ICNOutput with predictions and detailed analysis
        """
        batch_size = len(batch_input.input_ids_list)
        
        # Phase 1: Local Intent Estimation
        all_local_outputs = []
        all_local_embeddings = []
        all_local_intents = []
        max_units = 0
        
        for pkg_idx in range(batch_size):
            # Process each unit in the package
            pkg_input_ids = batch_input.input_ids_list[pkg_idx]
            pkg_attention_masks = batch_input.attention_masks_list[pkg_idx] 
            pkg_phase_ids = batch_input.phase_ids_list[pkg_idx]
            pkg_api_features = batch_input.api_features_list[pkg_idx]
            pkg_ast_features = batch_input.ast_features_list[pkg_idx]
            
            n_units = pkg_input_ids.shape[0]
            max_units = max(max_units, n_units)
            
            # Forward through local estimator
            local_output = self.local_estimator(
                input_ids=pkg_input_ids,
                attention_mask=pkg_attention_masks,
                phase_ids=pkg_phase_ids,
                api_features=pkg_api_features,
                ast_features=pkg_ast_features
            )
            
            all_local_outputs.append(local_output)
            all_local_embeddings.append(local_output.unit_embeddings)
            
            # Combine fixed and latent intents
            combined_intents = torch.cat([
                local_output.fixed_intent_dist,
                local_output.latent_intent_dist
            ], dim=-1)
            all_local_intents.append(combined_intents)
        
        # Phase 2: Pad and stack for batch processing
        local_embeddings_padded = torch.zeros(
            batch_size, max_units, self.embedding_dim
        ).to(all_local_embeddings[0].device)
        
        local_intents_padded = torch.zeros(
            batch_size, max_units, self.n_total_intents
        ).to(all_local_intents[0].device)
        
        unit_masks = torch.zeros(batch_size, max_units, dtype=torch.bool)
        
        for i, (embeddings, intents) in enumerate(zip(all_local_embeddings, all_local_intents)):
            n_units = embeddings.shape[0]
            local_embeddings_padded[i, :n_units] = embeddings
            local_intents_padded[i, :n_units] = intents
            unit_masks[i, :n_units] = True
        
        unit_masks = unit_masks.to(all_local_embeddings[0].device)
        
        # Phase 3: Global Intent Integration
        global_output = self.global_integrator(
            local_embeddings=local_embeddings_padded,
            local_intent_dists=local_intents_padded,
            unit_masks=unit_masks,
            manifest_embeddings=batch_input.manifest_embeddings,
            return_history=True
        )
        
        # Phase 4: Extract metrics for detection
        convergence_metrics = self.global_integrator.get_convergence_metrics(
            global_output.convergence_history
        )
        
        divergence_metrics = self.global_integrator.compute_divergence_metrics(
            local_intents_padded, global_output.final_global_intent, unit_masks
        )
        
        # Phase 5: Dual Detection
        # Check for phase violations (simplified)
        phase_violations = self._detect_phase_violations(
            all_local_outputs, batch_input.phase_ids_list
        )
        
        # Extract latent activations
        latent_activations = global_output.final_global_intent[:, self.n_fixed_intents:]
        
        detection_output = self.detection_system(
            convergence_metrics=convergence_metrics,
            divergence_metrics=divergence_metrics,
            local_embeddings=local_embeddings_padded,
            unit_masks=unit_masks,
            attention_weights=global_output.unit_attention_weights,
            global_embeddings=global_output.final_global_embedding,
            global_intent_dists=global_output.final_global_intent,
            phase_violations=phase_violations,
            latent_activations=latent_activations
        )
        
        return ICNOutput(
            malicious_scores=detection_output.final_scores,
            malicious_predictions=detection_output.final_predictions,
            local_outputs=all_local_outputs,
            global_output=global_output,
            detection_output=detection_output,
            convergence_metrics=convergence_metrics,
            divergence_metrics=divergence_metrics,
            explanations=detection_output.explanations
        )
    
    def _detect_phase_violations(
        self, 
        local_outputs: List[LocalIntentOutput],
        phase_ids_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Detect phase constraint violations (simplified version)."""
        violations = []
        
        for pkg_outputs, pkg_phases in zip(local_outputs, phase_ids_list):
            pkg_violations = 0
            
            for unit_output, phase_id in zip(pkg_outputs.fixed_intent_dist, pkg_phases):
                # Check for suspicious intents in install phase
                if phase_id.item() == 0:  # install phase
                    # Check for network, eval, proc.spawn in install
                    net_outbound = unit_output[0].item()  # net.outbound
                    proc_spawn = unit_output[4].item()    # proc.spawn
                    eval_intent = unit_output[5].item()   # eval
                    
                    if net_outbound > 0.3 or proc_spawn > 0.3 or eval_intent > 0.3:
                        pkg_violations += 1
            
            violations.append(pkg_violations)
        
        return torch.tensor(violations, dtype=torch.long)
    
    def interpret_predictions(self, output: ICNOutput, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Generate human-readable interpretations of predictions.
        
        Args:
            output: ICNOutput from forward pass
            top_k: Number of top intents/units to show
            
        Returns:
            List of interpretation dictionaries per sample
        """
        interpretations = []
        
        for i, explanation in enumerate(output.explanations):
            interpretation = {
                'package_index': i,
                'prediction': 'MALICIOUS' if output.malicious_predictions[i] else 'BENIGN',
                'confidence': output.malicious_scores[i].item(),
                'primary_detection_channel': explanation['primary_channel'],
                'convergence_info': {
                    'iterations': output.convergence_metrics.get('iterations_to_converge', 0),
                    'converged': output.convergence_metrics.get('converged', False),
                    'final_drift': output.convergence_metrics.get('final_drift', 0.0)
                }
            }
            
            # Add channel-specific details
            if explanation['primary_channel'] == 'divergence':
                div_evidence = explanation['divergence_channel']['evidence']
                interpretation['divergence_details'] = {
                    'suspicious_units_count': div_evidence['num_flagged_units'],
                    'max_divergence': div_evidence['max_divergence'],
                    'mean_divergence': div_evidence['mean_divergence']
                }
                
                # Show most suspicious units
                if div_evidence['suspicious_units']:
                    top_units = sorted(
                        div_evidence['suspicious_units'],
                        key=lambda x: x['malicious_prob'],
                        reverse=True
                    )[:top_k]
                    interpretation['most_suspicious_units'] = top_units
            
            else:  # plausibility channel
                plaus_evidence = explanation['plausibility_channel']['evidence']
                interpretation['plausibility_details'] = {
                    'distance_to_benign': plaus_evidence['min_distance_to_benign'],
                    'intent_entropy': plaus_evidence['intent_entropy'],
                    'phase_violations': plaus_evidence['phase_violations'],
                    'abnormal_intents': plaus_evidence['abnormal_intents']
                }
            
            # Add global intent interpretation
            global_intents = output.global_output.final_global_intent[i]
            top_intent_indices = torch.argsort(global_intents, descending=True)[:top_k]
            
            top_intents = []
            for idx in top_intent_indices:
                if idx < self.n_fixed_intents:
                    intent_name = self.intent_vocab.idx_to_intent[idx.item()]
                else:
                    intent_name = f"latent_{idx.item() - self.n_fixed_intents}"
                
                top_intents.append({
                    'intent': intent_name,
                    'activation': global_intents[idx].item()
                })
            
            interpretation['top_global_intents'] = top_intents
            interpretations.append(interpretation)
        
        return interpretations
    
    def fit_benign_manifold(self, benign_embeddings: torch.Tensor):
        """Fit the benign manifold for plausibility detection."""
        self.detection_system.plausibility_detector.fit_benign_manifold(benign_embeddings)
    
    def set_training_mode(self, mode: bool = True):
        """Set training mode for the model."""
        self.training_mode = mode
        self.train(mode)


if __name__ == "__main__":
    # Test the complete ICN model
    print("Testing Complete ICN Model...")
    
    # Create model
    model = ICNModel(
        embedding_dim=256,
        hidden_dim=128,
        use_pretrained=False,  # Use custom for testing
        max_iterations=3
    )
    
    batch_size = 2
    max_seq_len = 64
    
    # Create dummy batch input
    input_ids_list = []
    attention_masks_list = []
    phase_ids_list = []
    api_features_list = []
    ast_features_list = []
    
    for i in range(batch_size):
        n_units = 3 + i  # Variable number of units per package
        
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
        sample_types=['benign', 'compromised_lib'],
        malicious_labels=torch.tensor([0, 1], dtype=torch.float)
    )
    
    # Fit benign manifold (required)
    benign_embeddings = torch.randn(50, 256)
    model.fit_benign_manifold(benign_embeddings)
    
    # Forward pass
    print("Running forward pass...")
    output = model(batch_input)
    
    print(f"Malicious scores: {output.malicious_scores}")
    print(f"Predictions: {output.malicious_predictions}")
    print(f"Converged: {output.global_output.converged}")
    print(f"Final iteration: {output.global_output.final_iteration}")
    
    # Test interpretations
    print(f"\nGenerating interpretations...")
    interpretations = model.interpret_predictions(output)
    
    for i, interp in enumerate(interpretations):
        print(f"\nPackage {i}:")
        print(f"  Prediction: {interp['prediction']}")
        print(f"  Confidence: {interp['confidence']:.3f}")
        print(f"  Primary channel: {interp['primary_detection_channel']}")
        print(f"  Convergence: {interp['convergence_info']}")
        intent_strs = [f"{intent['intent']}({intent['activation']:.2f})" for intent in interp['top_global_intents']]
        print(f"  Top intents: {intent_strs}")
    
    print("\n✅ Complete ICN Model implemented successfully!")
    print("Ready for training and evaluation!")