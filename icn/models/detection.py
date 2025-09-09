"""
Dual Detection Channels for ICN.
Implements divergence and plausibility detection for different types of malicious packages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class DetectionChannel(Enum):
    """Types of detection channels."""
    DIVERGENCE = "divergence"      # For compromised_lib (trojanized packages)
    PLAUSIBILITY = "plausibility"  # For malicious_intent (fully malicious packages)


@dataclass
class DetectionResult:
    """Result from a detection channel."""
    score: float  # 0-1 maliciousness score
    evidence: Dict  # Supporting evidence for the detection
    channel: DetectionChannel
    confidence: float  # Confidence in the detection


class DualDetectionOutput(NamedTuple):
    """Combined output from both detection channels."""
    divergence_results: List[DetectionResult]
    plausibility_results: List[DetectionResult] 
    final_scores: torch.Tensor  # [batch_size] combined scores
    final_predictions: torch.Tensor  # [batch_size] binary predictions
    explanations: List[Dict]  # Explanations per sample


class DivergenceDetector(nn.Module):
    """
    Divergence channel for detecting compromised_lib samples (trojans).
    Focuses on local-global intent misalignment and convergence failures.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        divergence_threshold: float = 0.5,
        max_units_to_flag: int = 5
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.divergence_threshold = divergence_threshold
        self.max_units_to_flag = max_units_to_flag
        
        # Divergence scoring network
        self.divergence_scorer = nn.Sequential(
            nn.Linear(4, hidden_dim),  # mean_div, max_div, iterations, drift
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Unit-level anomaly detector
        self.unit_anomaly_detector = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),  # embedding + divergence
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),  # malicious vs benign unit
        )
        
    def forward(
        self,
        convergence_metrics: Dict[str, float],
        divergence_metrics: Dict[str, torch.Tensor],
        local_embeddings: torch.Tensor,
        unit_masks: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> List[DetectionResult]:
        """
        Detect trojans using divergence patterns.
        
        Args:
            convergence_metrics: Metrics about convergence speed/quality
            divergence_metrics: KL divergences between locals and global
            local_embeddings: Local unit embeddings
            unit_masks: Valid unit masks
            attention_weights: Attention weights from global integrator
            
        Returns:
            List of DetectionResult objects
        """
        batch_size = local_embeddings.shape[0]
        results = []
        
        mean_divergence = divergence_metrics['mean_divergence']
        max_divergence = divergence_metrics['max_divergence']
        per_unit_divergence = divergence_metrics['per_unit_divergence']
        
        for i in range(batch_size):
            # Extract features for divergence scoring
            features = torch.tensor([
                mean_divergence[i].item(),
                max_divergence[i].item(),
                convergence_metrics.get('iterations_to_converge', 3),
                convergence_metrics.get('final_drift', 0.01)
            ], device=local_embeddings.device)
            
            # Compute divergence score
            div_score = self.divergence_scorer(features).item()
            
            # Identify suspicious units
            unit_divergences = per_unit_divergence[i]
            unit_mask = unit_masks[i]
            
            # Find units with high divergence
            masked_divergences = unit_divergences * unit_mask.float()
            top_divergent_indices = torch.argsort(
                masked_divergences, descending=True
            )[:self.max_units_to_flag]
            
            suspicious_units = []
            for unit_idx in top_divergent_indices:
                if unit_mask[unit_idx] and masked_divergences[unit_idx] > self.divergence_threshold:
                    # Classify unit as malicious/benign
                    unit_features = torch.cat([
                        local_embeddings[i, unit_idx],
                        unit_divergences[unit_idx].unsqueeze(0)
                    ])
                    
                    unit_logits = self.unit_anomaly_detector(unit_features)
                    unit_prob = F.softmax(unit_logits, dim=0)[1].item()  # malicious probability
                    
                    suspicious_units.append({
                        'unit_index': unit_idx.item(),
                        'divergence': masked_divergences[unit_idx].item(),
                        'malicious_prob': unit_prob,
                        'attention_weight': attention_weights[i, unit_idx].item()
                    })
            
            # Prepare evidence
            evidence = {
                'mean_divergence': mean_divergence[i].item(),
                'max_divergence': max_divergence[i].item(),
                'convergence_iterations': convergence_metrics.get('iterations_to_converge', 3),
                'convergence_drift': convergence_metrics.get('final_drift', 0.01),
                'suspicious_units': suspicious_units,
                'num_flagged_units': len(suspicious_units)
            }
            
            # Compute confidence based on consistency of signals
            confidence = self._compute_divergence_confidence(evidence)
            
            result = DetectionResult(
                score=div_score,
                evidence=evidence,
                channel=DetectionChannel.DIVERGENCE,
                confidence=confidence
            )
            results.append(result)
        
        return results
    
    def _compute_divergence_confidence(self, evidence: Dict) -> float:
        """Compute confidence in divergence detection."""
        # High confidence when:
        # - Clear separation between max and mean divergence (one bad unit)
        # - Slow convergence
        # - High-confidence malicious unit classifications
        
        max_div = evidence['max_divergence']
        mean_div = evidence['mean_divergence']
        
        # Separation score
        separation = (max_div - mean_div) / (max_div + 1e-8)
        
        # Convergence penalty
        conv_penalty = min(evidence['convergence_iterations'] / 6.0, 1.0)
        
        # Unit classification confidence
        unit_confidences = [
            unit['malicious_prob'] for unit in evidence['suspicious_units']
        ]
        avg_unit_confidence = np.mean(unit_confidences) if unit_confidences else 0.5
        
        # Combined confidence
        confidence = 0.4 * separation + 0.3 * conv_penalty + 0.3 * avg_unit_confidence
        return min(max(confidence, 0.0), 1.0)


class PlausibilityDetector(nn.Module):
    """
    Plausibility channel for detecting malicious_intent samples.
    Focuses on global intent abnormality and phase constraint violations.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        n_benign_prototypes: int = 10,
        hidden_dim: int = 256,
        plausibility_threshold: float = 0.3
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.n_benign_prototypes = n_benign_prototypes
        self.plausibility_threshold = plausibility_threshold
        
        # Benign manifold (will be fitted during training)
        self.register_buffer(
            'benign_prototypes', 
            torch.randn(n_benign_prototypes, embedding_dim)
        )
        self.manifold_fitted = False
        
        # Plausibility scoring network
        self.plausibility_scorer = nn.Sequential(
            nn.Linear(6, hidden_dim),  # distance metrics + phase features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Phase constraint checker
        self.phase_constraints = {
            # Constraints: (intent_category, allowed_phases)
            'net.outbound': {'runtime'},  # Network calls should be runtime
            'proc.spawn': {'runtime'},    # Process spawning should be runtime
            'eval': {'runtime'},          # Code evaluation should be runtime  
            'sys.env': {'install', 'runtime'},  # Environment access more flexible
        }
        
    def fit_benign_manifold(self, benign_embeddings: torch.Tensor):
        """Fit the benign manifold using K-means clustering."""
        if benign_embeddings.shape[0] < self.n_benign_prototypes:
            # Not enough samples, use what we have
            n_actual = benign_embeddings.shape[0]
            self.benign_prototypes[:n_actual] = benign_embeddings
            self.benign_prototypes[n_actual:] = benign_embeddings[
                torch.randint(0, n_actual, (self.n_benign_prototypes - n_actual,))
            ]
        else:
            # Use K-means or simple sampling
            try:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=self.n_benign_prototypes, random_state=42)
                centers = kmeans.fit(benign_embeddings.cpu().numpy()).cluster_centers_
                self.benign_prototypes = torch.from_numpy(centers).float().to(
                    benign_embeddings.device
                )
            except ImportError:
                # Fallback: random sampling
                indices = torch.randperm(benign_embeddings.shape[0])[:self.n_benign_prototypes]
                self.benign_prototypes = benign_embeddings[indices]
        
        self.manifold_fitted = True
    
    def forward(
        self,
        global_embeddings: torch.Tensor,
        global_intent_dists: torch.Tensor,
        phase_violations: Optional[torch.Tensor] = None,
        latent_activations: Optional[torch.Tensor] = None
    ) -> List[DetectionResult]:
        """
        Detect malicious-by-design packages using plausibility analysis.
        
        Args:
            global_embeddings: Global package embeddings
            global_intent_dists: Global intent distributions
            phase_violations: Binary indicators of phase constraint violations
            latent_activations: Latent intent activation levels
            
        Returns:
            List of DetectionResult objects
        """
        if not self.manifold_fitted:
            # During manifold fitting phase, return dummy results
            batch_size = global_embeddings.shape[0] if global_embeddings is not None else 1
            dummy_results = []
            for i in range(batch_size):
                dummy_results.append(DetectionResult(
                    score=0.5,
                    evidence={"status": "manifold_not_fitted"},
                    channel=DetectionChannel.PLAUSIBILITY,
                    confidence=0.0
                ))
            return dummy_results
        
        batch_size = global_embeddings.shape[0]
        results = []
        
        for i in range(batch_size):
            global_embed = global_embeddings[i]
            global_intent = global_intent_dists[i]
            
            # Compute distance to benign manifold
            distances = torch.norm(
                global_embed.unsqueeze(0) - self.benign_prototypes,
                dim=-1
            )
            min_distance = torch.min(distances).item()
            mean_distance = torch.mean(distances).item()
            std_distance = torch.std(distances).item()
            
            # Analyze intent distribution abnormalities
            intent_entropy = self._compute_intent_entropy(global_intent)
            intent_concentration = torch.max(global_intent).item()
            
            # Check phase violations
            phase_violation_count = 0
            if phase_violations is not None:
                phase_violation_count = phase_violations[i].sum().item()
            
            # Check latent overactivation
            latent_overactivation = 0
            if latent_activations is not None:
                # Check if latent slots dominate (>0.5 of total probability)
                latent_overactivation = (latent_activations[i] > 0.1).sum().item()
            
            # Prepare features for plausibility scoring
            features = torch.tensor([
                min_distance,
                mean_distance,
                intent_entropy,
                intent_concentration,
                phase_violation_count,
                latent_overactivation
            ], device=global_embeddings.device)
            
            # Compute plausibility score
            plausibility_score = self.plausibility_scorer(features).item()
            
            # Prepare evidence
            evidence = {
                'min_distance_to_benign': min_distance,
                'mean_distance_to_benign': mean_distance,
                'distance_std': std_distance,
                'intent_entropy': intent_entropy,
                'intent_concentration': intent_concentration,
                'phase_violations': phase_violation_count,
                'latent_overactivation': latent_overactivation,
                'abnormal_intents': self._identify_abnormal_intents(global_intent)
            }
            
            # Compute confidence
            confidence = self._compute_plausibility_confidence(evidence)
            
            result = DetectionResult(
                score=plausibility_score,
                evidence=evidence,
                channel=DetectionChannel.PLAUSIBILITY,
                confidence=confidence
            )
            results.append(result)
        
        return results
    
    def _compute_intent_entropy(self, intent_dist: torch.Tensor) -> float:
        """Compute entropy of intent distribution."""
        eps = 1e-8
        entropy = -(intent_dist * torch.log(intent_dist + eps)).sum().item()
        return entropy
    
    def _identify_abnormal_intents(
        self, 
        intent_dist: torch.Tensor,
        threshold: float = 0.1
    ) -> List[int]:
        """Identify intent categories with unusually high activation."""
        high_intents = (intent_dist > threshold).nonzero(as_tuple=True)[0]
        return high_intents.tolist()
    
    def _compute_plausibility_confidence(self, evidence: Dict) -> float:
        """Compute confidence in plausibility detection."""
        # High confidence when:
        # - Large distance from benign manifold
        # - Multiple types of abnormalities (intents + phase + latent)
        # - Consistent abnormal patterns
        
        distance_score = min(evidence['min_distance_to_benign'] / 2.0, 1.0)
        
        # Count different types of abnormalities
        abnormality_types = 0
        if len(evidence['abnormal_intents']) > 3:
            abnormality_types += 1
        if evidence['phase_violations'] > 0:
            abnormality_types += 1
        if evidence['latent_overactivation'] > 2:
            abnormality_types += 1
        
        diversity_score = abnormality_types / 3.0
        
        # Entropy-based confidence (very low or very high entropy is suspicious)
        entropy = evidence['intent_entropy']
        entropy_score = 1.0 - abs(entropy - 1.5) / 1.5  # Normalized around expected entropy
        
        confidence = 0.5 * distance_score + 0.3 * diversity_score + 0.2 * entropy_score
        return min(max(confidence, 0.0), 1.0)
    
    def get_prototypes(self) -> torch.Tensor:
        """Return the benign prototypes for loss computation."""
        if not self.manifold_fitted:
            # Return dummy prototypes if manifold not fitted yet
            return torch.zeros_like(self.benign_prototypes)
        return self.benign_prototypes


class DualDetectionSystem(nn.Module):
    """
    Combined dual detection system that integrates divergence and plausibility channels.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        divergence_weight: float = 0.6,
        plausibility_weight: float = 0.4,
        final_threshold: float = 0.5
    ):
        super().__init__()
        
        self.divergence_detector = DivergenceDetector(embedding_dim)
        self.plausibility_detector = PlausibilityDetector(embedding_dim)
        
        self.divergence_weight = divergence_weight
        self.plausibility_weight = plausibility_weight
        self.final_threshold = final_threshold
        
        # Final decision fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(4, 64),  # div_score, plaus_score, div_conf, plaus_conf
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        # Divergence channel inputs
        convergence_metrics: Dict[str, float],
        divergence_metrics: Dict[str, torch.Tensor],
        local_embeddings: torch.Tensor,
        unit_masks: torch.Tensor,
        attention_weights: torch.Tensor,
        # Plausibility channel inputs
        global_embeddings: torch.Tensor,
        global_intent_dists: torch.Tensor,
        phase_violations: Optional[torch.Tensor] = None,
        latent_activations: Optional[torch.Tensor] = None
    ) -> DualDetectionOutput:
        """
        Run both detection channels and combine results.
        
        Returns:
            DualDetectionOutput with combined detection results
        """
        # Run divergence detection
        divergence_results = self.divergence_detector(
            convergence_metrics=convergence_metrics,
            divergence_metrics=divergence_metrics,
            local_embeddings=local_embeddings,
            unit_masks=unit_masks,
            attention_weights=attention_weights
        )
        
        # Run plausibility detection  
        plausibility_results = self.plausibility_detector(
            global_embeddings=global_embeddings,
            global_intent_dists=global_intent_dists,
            phase_violations=phase_violations,
            latent_activations=latent_activations
        )
        
        # Combine results
        batch_size = len(divergence_results)
        final_scores = []
        explanations = []
        
        for i in range(batch_size):
            div_result = divergence_results[i]
            plaus_result = plausibility_results[i]
            
            # Prepare fusion features
            fusion_features = torch.tensor([
                div_result.score,
                plaus_result.score,
                div_result.confidence,
                plaus_result.confidence
            ], device=global_embeddings.device)
            
            # Fused score
            fused_score = self.fusion_network(fusion_features).item()
            final_scores.append(fused_score)
            
            # Create explanation
            explanation = {
                'final_score': fused_score,
                'divergence_channel': {
                    'score': div_result.score,
                    'confidence': div_result.confidence,
                    'evidence': div_result.evidence
                },
                'plausibility_channel': {
                    'score': plaus_result.score,
                    'confidence': plaus_result.confidence,
                    'evidence': plaus_result.evidence
                },
                'primary_channel': 'divergence' if div_result.score * div_result.confidence > 
                                 plaus_result.score * plaus_result.confidence else 'plausibility'
            }
            explanations.append(explanation)
        
        final_scores_tensor = torch.tensor(final_scores, device=global_embeddings.device)
        predictions = (final_scores_tensor > self.final_threshold).long()
        
        return DualDetectionOutput(
            divergence_results=divergence_results,
            plausibility_results=plausibility_results,
            final_scores=final_scores_tensor,
            final_predictions=predictions,
            explanations=explanations
        )


if __name__ == "__main__":
    # Test the dual detection system
    print("Testing Dual Detection System...")
    
    batch_size = 4
    max_units = 6
    embedding_dim = 256
    n_intents = 25
    
    # Create dummy inputs
    local_embeddings = torch.randn(batch_size, max_units, embedding_dim)
    global_embeddings = torch.randn(batch_size, embedding_dim)
    global_intent_dists = F.softmax(torch.randn(batch_size, n_intents), dim=-1)
    unit_masks = torch.randint(0, 2, (batch_size, max_units)).bool()
    attention_weights = F.softmax(torch.randn(batch_size, max_units), dim=-1)
    
    # Create dummy metrics
    convergence_metrics = {
        'iterations_to_converge': 3,
        'final_drift': 0.02,
        'converged': True
    }
    
    divergence_metrics = {
        'mean_divergence': torch.rand(batch_size),
        'max_divergence': torch.rand(batch_size) + 0.5,  # Higher max divergence
        'per_unit_divergence': torch.rand(batch_size, max_units)
    }
    
    phase_violations = torch.randint(0, 3, (batch_size,))
    latent_activations = torch.rand(batch_size, 10)
    
    # Create detection system
    detector = DualDetectionSystem(embedding_dim=embedding_dim)
    
    # Fit benign manifold (required for plausibility detector)
    benign_samples = torch.randn(100, embedding_dim)  # Mock benign embeddings
    detector.plausibility_detector.fit_benign_manifold(benign_samples)
    
    # Run detection
    output = detector(
        convergence_metrics=convergence_metrics,
        divergence_metrics=divergence_metrics,
        local_embeddings=local_embeddings,
        unit_masks=unit_masks,
        attention_weights=attention_weights,
        global_embeddings=global_embeddings,
        global_intent_dists=global_intent_dists,
        phase_violations=phase_violations,
        latent_activations=latent_activations
    )
    
    print(f"Final scores: {output.final_scores}")
    print(f"Predictions: {output.final_predictions}")
    
    # Show detailed results for first sample
    print(f"\nDetailed results for sample 0:")
    explanation = output.explanations[0]
    print(f"  Final score: {explanation['final_score']:.3f}")
    print(f"  Primary channel: {explanation['primary_channel']}")
    print(f"  Divergence: score={explanation['divergence_channel']['score']:.3f}, "
          f"conf={explanation['divergence_channel']['confidence']:.3f}")
    print(f"  Plausibility: score={explanation['plausibility_channel']['score']:.3f}, "
          f"conf={explanation['plausibility_channel']['confidence']:.3f}")
    
    # Show suspicious units from divergence channel
    div_evidence = explanation['divergence_channel']['evidence']
    if div_evidence['suspicious_units']:
        print(f"  Suspicious units: {len(div_evidence['suspicious_units'])}")
        for unit in div_evidence['suspicious_units'][:2]:  # Show first 2
            print(f"    Unit {unit['unit_index']}: div={unit['divergence']:.3f}, "
                  f"malicious_prob={unit['malicious_prob']:.3f}")
    
    print("\n✅ Dual Detection System implemented successfully!")