"""
ICN Training Losses implementing the specifications from training.md.

Maps dataset categories to appropriate loss functions:
- compromised_lib → Divergence Margin Loss (divergence channel)
- malicious_intent → Global Plausibility Loss (plausibility channel)  
- benign → Convergence Loss (stable alignment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math


class SampleType(Enum):
    """Dataset sample categories."""
    BENIGN = "benign"
    COMPROMISED_LIB = "compromised_lib"  # Trojans - divergence channel
    MALICIOUS_INTENT = "malicious_intent"  # Malicious-by-design - plausibility channel


class ICNLossComputer:
    """Computes ICN training losses based on sample types and convergence states."""
    
    def __init__(
        self,
        n_fixed_intents: int = 15,
        n_latent_intents: int = 10,
        divergence_margin: float = 1.0,
        plausibility_margin: float = 0.5,
        temperature: float = 0.07
    ):
        self.n_fixed_intents = n_fixed_intents
        self.n_latent_intents = n_latent_intents
        self.divergence_margin = divergence_margin
        self.plausibility_margin = plausibility_margin
        self.temperature = temperature
        
        # Loss weights (will be tuned during training)
        self.loss_weights = {
            'intent_supervision': 1.0,
            'convergence': 1.0,
            'divergence_margin': 2.0,
            'plausibility': 2.0,
            'classification': 1.0,
            'latent_contrastive': 0.5
        }
    
    def compute_losses(
        self,
        local_outputs: List[torch.Tensor],  # List of LocalIntentOutput per unit
        global_output: torch.Tensor,  # Global intent distribution [batch_size, n_intents]
        global_embeddings: torch.Tensor,  # Global embeddings [batch_size, embedding_dim]
        sample_types: List[SampleType],  # Sample type per batch item
        intent_labels: Optional[torch.Tensor] = None,  # Weak supervision labels
        malicious_labels: Optional[torch.Tensor] = None,  # Binary classification labels
        benign_manifold: Optional[torch.Tensor] = None,  # Benign embedding prototypes
        convergence_history: Optional[List[torch.Tensor]] = None,  # History of global states
        phase_constraints: Optional[torch.Tensor] = None  # Phase constraint violations
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all ICN losses based on sample types.
        
        Args:
            local_outputs: Local intent outputs for each unit in each package
            global_output: Global intent distributions 
            global_embeddings: Global package embeddings
            sample_types: Type of each sample (benign, compromised_lib, malicious_intent)
            intent_labels: Weak supervision labels for fixed intents
            malicious_labels: Binary malicious/benign labels
            benign_manifold: Prototype embeddings for benign packages
            convergence_history: Evolution of global states during convergence
            phase_constraints: Phase constraint violation indicators
        
        Returns:
            Dictionary of computed losses
        """
        batch_size = len(sample_types)
        losses = {}
        
        # 1. Intent Supervision Loss (all samples)
        if intent_labels is not None:
            losses['intent_supervision'] = self._compute_intent_supervision_loss(
                local_outputs, intent_labels
            )
        
        # 2. Sample-type specific losses
        benign_indices = [i for i, t in enumerate(sample_types) if t == SampleType.BENIGN]
        compromised_indices = [i for i, t in enumerate(sample_types) if t == SampleType.COMPROMISED_LIB]
        malicious_indices = [i for i, t in enumerate(sample_types) if t == SampleType.MALICIOUS_INTENT]
        
        # 2a. Convergence Loss (benign samples)
        if benign_indices and convergence_history is not None:
            losses['convergence'] = self._compute_convergence_loss(
                local_outputs, global_output, convergence_history, benign_indices
            )
        
        # 2b. Divergence Margin Loss (compromised_lib samples)
        if compromised_indices:
            losses['divergence_margin'] = self._compute_divergence_margin_loss(
                local_outputs, global_output, compromised_indices
            )
        
        # 2c. Global Plausibility Loss (malicious_intent samples)
        if malicious_indices and benign_manifold is not None:
            losses['plausibility'] = self._compute_plausibility_loss(
                global_embeddings, benign_manifold, malicious_indices, phase_constraints
            )
        
        # 3. Final Classification Loss
        if malicious_labels is not None:
            losses['classification'] = self._compute_classification_loss(
                global_embeddings, malicious_labels
            )
        
        # 4. Latent Intent Discovery Loss
        losses['latent_contrastive'] = self._compute_latent_contrastive_loss(
            local_outputs
        )
        
        # 5. Compute weighted total loss
        total_loss = torch.tensor(0.0, device=global_embeddings.device)
        for loss_name, loss_value in losses.items():
            if loss_value is not None:
                weighted_loss = self.loss_weights[loss_name] * loss_value
                total_loss += weighted_loss
                losses[f'{loss_name}_weighted'] = weighted_loss
        
        losses['total'] = total_loss
        return losses
    
    def _compute_intent_supervision_loss(
        self, 
        local_outputs: List[torch.Tensor], 
        intent_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Weak supervision on fixed intents using API call mappings.
        Uses binary cross entropy with label smoothing.
        """
        batch_losses = []
        
        for i, local_output in enumerate(local_outputs):
            # local_output is a LocalIntentOutput with batch dimension for units
            # fixed_intent_dist is [n_units, n_fixed_intents]
            fixed_dist = local_output.fixed_intent_dist
            target = intent_labels[i].float().unsqueeze(0).expand(fixed_dist.shape[0], -1)
            
            # Binary cross entropy with logits (safe for autocast)
            # Convert probabilities back to logits for safe mixed precision
            fixed_logits = torch.logit(torch.clamp(fixed_dist, min=1e-7, max=1-1e-7))
            loss = F.binary_cross_entropy_with_logits(fixed_logits, target, reduction='mean')
            batch_losses.append(loss)
        
        return torch.stack(batch_losses).mean() if batch_losses else torch.tensor(0.0)
    
    def _compute_convergence_loss(
        self,
        local_outputs: List[torch.Tensor],
        global_output: torch.Tensor,
        convergence_history: List[torch.Tensor],
        benign_indices: List[int]
    ) -> torch.Tensor:
        """
        Encourage rapid convergence for benign samples.
        Penalize packages that require many iterations to converge.
        """
        if not benign_indices:
            return torch.tensor(0.0)
        
        convergence_losses = []
        
        for i in benign_indices:
            local_output = local_outputs[i]
            global_dist = global_output[i]
            
            # Combine fixed and latent distributions for all units
            # local_output.fixed_intent_dist: [n_units, n_fixed_intents]
            # local_output.latent_intent_dist: [n_units, n_latent_intents]
            local_combined = torch.cat([
                local_output.fixed_intent_dist,
                local_output.latent_intent_dist
            ], dim=-1)  # [n_units, n_total_intents]
            
            # Expand global distribution to match units
            global_expanded = global_dist.unsqueeze(0).expand(local_combined.shape[0], -1)
            
            # KL divergence for each unit: KL(local || global)
            kl_div = F.kl_div(
                torch.log(local_combined + 1e-8),
                global_expanded,
                reduction='none'
            ).sum(dim=-1)  # [n_units]
            
            # Average divergence across units (should be small for benign)
            avg_divergence = kl_div.mean()
            convergence_losses.append(avg_divergence)
        
        # Also penalize slow convergence (if convergence took many iterations)
        if len(convergence_history) > 3:  # More than 3 iterations = slow
            slow_convergence_penalty = torch.tensor(
                len(convergence_history) - 3, 
                dtype=torch.float32,
                device=global_output.device
            ) * 0.1
            return torch.stack(convergence_losses).mean() + slow_convergence_penalty
        
        return torch.stack(convergence_losses).mean()
    
    def _compute_divergence_margin_loss(
        self,
        local_outputs: List[torch.Tensor],
        global_output: torch.Tensor, 
        compromised_indices: List[int]
    ) -> torch.Tensor:
        """
        Divergence Margin Loss for compromised_lib samples.
        Enforce that at least one local unit diverges from global intent.
        """
        if not compromised_indices:
            return torch.tensor(0.0)
        
        margin_losses = []
        
        for i in compromised_indices:
            local_output = local_outputs[i]
            global_dist = global_output[i]
            
            # Combine fixed and latent distributions for all units
            local_combined = torch.cat([
                local_output.fixed_intent_dist,
                local_output.latent_intent_dist
            ], dim=-1)  # [n_units, n_total_intents]
            
            # Expand global distribution to match units
            global_expanded = global_dist.unsqueeze(0).expand(local_combined.shape[0], -1)
            
            # KL divergence for each unit: KL(local || global)
            kl_divergences = F.kl_div(
                torch.log(local_combined + 1e-8),
                global_expanded,
                reduction='none'
            ).sum(dim=-1)  # [n_units]
            
            # Find maximum divergence (the malicious unit should have high divergence)
            max_divergence = torch.max(kl_divergences)
            
            # Margin loss: max(0, margin - max_divergence)
            # We want max_divergence to be at least 'margin'
            margin_loss = F.relu(self.divergence_margin - max_divergence)
            margin_losses.append(margin_loss)
        
        return torch.stack(margin_losses).mean()
    
    def _compute_plausibility_loss(
        self,
        global_embeddings: torch.Tensor,
        benign_manifold: torch.Tensor,
        malicious_indices: List[int],
        phase_constraints: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Global Plausibility Loss for malicious_intent samples.
        Push malicious global embeddings away from benign manifold.
        """
        if not malicious_indices:
            return torch.tensor(0.0)
        
        plausibility_losses = []
        
        # Compute distances to benign manifold
        for i in malicious_indices:
            global_embed = global_embeddings[i]
            
            # Distance to nearest benign prototype
            distances = torch.norm(
                global_embed.unsqueeze(0) - benign_manifold, 
                dim=-1
            )
            min_distance = torch.min(distances)
            
            # Margin loss: max(0, margin - distance)
            # We want distance to be at least 'margin'
            plausibility_loss = F.relu(self.plausibility_margin - min_distance)
            plausibility_losses.append(plausibility_loss)
        
        base_loss = torch.stack(plausibility_losses).mean()
        
        # Add phase constraint violations if available
        if phase_constraints is not None:
            phase_penalty = phase_constraints[malicious_indices].float().mean()
            return base_loss + 0.5 * phase_penalty
        
        return base_loss
    
    def _compute_classification_loss(
        self,
        global_embeddings: torch.Tensor,
        malicious_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Binary classification loss for final malicious/benign prediction.
        Uses a simple linear classifier on global embeddings.
        """
        # Simple linear classifier (could be made more sophisticated)
        logits = torch.sum(global_embeddings, dim=-1)  # Simple aggregation
        
        # Binary cross entropy
        return F.binary_cross_entropy_with_logits(
            logits, malicious_labels.float()
        )
    
    def _compute_latent_contrastive_loss(
        self,
        local_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Contrastive learning loss for latent intent discovery.
        Encourages similar units to have similar latent activations.
        """
        all_latent_dists = []
        all_embeddings = []
        
        # Collect all latent distributions and embeddings
        for local_output in local_outputs:
            # local_output is a LocalIntentOutput with batch dimensions for units
            n_units = local_output.latent_intent_dist.shape[0]
            for unit_idx in range(n_units):
                all_latent_dists.append(local_output.latent_intent_dist[unit_idx])
                all_embeddings.append(local_output.unit_embeddings[unit_idx])
        
        if len(all_latent_dists) < 2:
            return torch.tensor(0.0)
        
        latent_dists = torch.stack(all_latent_dists)
        embeddings = torch.stack(all_embeddings)
        
        # Compute pairwise similarities
        norm_embeddings = F.normalize(embeddings, dim=-1)
        similarity_matrix = torch.matmul(norm_embeddings, norm_embeddings.T)
        
        # Contrastive loss based on latent distribution similarity
        latent_similarity = F.cosine_similarity(
            latent_dists.unsqueeze(1), 
            latent_dists.unsqueeze(0),
            dim=-1
        )
        
        # InfoNCE-style contrastive loss
        pos_pairs = similarity_matrix > 0.7  # High embedding similarity
        neg_pairs = similarity_matrix < 0.3  # Low embedding similarity
        
        pos_loss = -torch.log(latent_similarity + 1e-8)[pos_pairs].mean()
        neg_loss = torch.log(1 - latent_similarity + 1e-8)[neg_pairs].mean()
        
        if torch.isnan(pos_loss):
            pos_loss = torch.tensor(0.0, device=embeddings.device)
        if torch.isnan(neg_loss):
            neg_loss = torch.tensor(0.0, device=embeddings.device)
        
        return pos_loss + neg_loss


class BenignManifoldModel:
    """
    Models the benign package embedding manifold for plausibility loss.
    Uses a simple density estimation approach.
    """
    
    def __init__(self, embedding_dim: int, n_components: int = 10):
        self.embedding_dim = embedding_dim
        self.n_components = n_components
        self.prototypes = None
        self.fitted = False
    
    def fit(self, benign_embeddings: torch.Tensor):
        """Fit the benign manifold using K-means clustering."""
        try:
            from sklearn.cluster import KMeans
            
            # Convert to numpy for sklearn
            embeddings_np = benign_embeddings.cpu().numpy()
            
            # Fit K-means
            kmeans = KMeans(n_clusters=self.n_components, random_state=42)
            kmeans.fit(embeddings_np)
            
            # Store cluster centers as prototypes
            self.prototypes = torch.from_numpy(kmeans.cluster_centers_).float()
            self.fitted = True
            
        except ImportError:
            # Fallback: use random sampling of benign embeddings
            indices = torch.randperm(benign_embeddings.size(0))[:self.n_components]
            self.prototypes = benign_embeddings[indices].clone()
            self.fitted = True
    
    def get_prototypes(self, device: torch.device) -> torch.Tensor:
        """Get prototype embeddings on specified device."""
        if not self.fitted:
            raise ValueError("Manifold model not fitted yet")
        return self.prototypes.to(device)
    
    def compute_anomaly_score(
        self, 
        embeddings: torch.Tensor, 
        device: torch.device
    ) -> torch.Tensor:
        """Compute anomaly score (distance to nearest prototype)."""
        prototypes = self.get_prototypes(device)
        
        # Compute distances to all prototypes
        distances = torch.cdist(embeddings, prototypes)
        
        # Return minimum distance (distance to nearest prototype)
        min_distances = torch.min(distances, dim=-1)[0]
        return min_distances


if __name__ == "__main__":
    # Test the loss computation
    print("Testing ICN Loss Computation...")
    
    # Create dummy data
    batch_size = 8
    embedding_dim = 256
    n_units_per_package = [3, 2, 4, 1, 3, 2, 3, 2]  # Variable units per package
    
    # Create dummy local outputs
    from icn.models.local_estimator import LocalIntentOutput
    
    local_outputs = []
    for i in range(batch_size):
        units = []
        for j in range(n_units_per_package[i]):
            unit_output = LocalIntentOutput(
                fixed_intent_dist=F.softmax(torch.randn(15), dim=-1),
                latent_intent_dist=F.softmax(torch.randn(10), dim=-1),
                unit_embeddings=torch.randn(embedding_dim)
            )
            units.append(unit_output)
        local_outputs.append(units)
    
    # Create dummy global outputs
    global_output = F.softmax(torch.randn(batch_size, 25), dim=-1)  # 15 fixed + 10 latent
    global_embeddings = torch.randn(batch_size, embedding_dim)
    
    # Create sample types
    sample_types = [
        SampleType.BENIGN, SampleType.BENIGN, SampleType.COMPROMISED_LIB,
        SampleType.MALICIOUS_INTENT, SampleType.BENIGN, SampleType.COMPROMISED_LIB,
        SampleType.MALICIOUS_INTENT, SampleType.BENIGN
    ]
    
    # Create dummy labels and constraints
    intent_labels = torch.randint(0, 2, (batch_size, 15)).float()
    malicious_labels = torch.tensor([0, 0, 1, 1, 0, 1, 1, 0]).float()
    
    # Create benign manifold
    manifold = BenignManifoldModel(embedding_dim)
    benign_embeddings = torch.randn(50, embedding_dim)  # 50 benign samples for training
    manifold.fit(benign_embeddings)
    
    # Create loss computer
    loss_computer = ICNLossComputer()
    
    # Compute losses
    losses = loss_computer.compute_losses(
        local_outputs=local_outputs,
        global_output=global_output,
        global_embeddings=global_embeddings,
        sample_types=sample_types,
        intent_labels=intent_labels,
        malicious_labels=malicious_labels,
        benign_manifold=manifold.get_prototypes(torch.device('cpu')),
        convergence_history=[torch.randn(batch_size, 25) for _ in range(2)]
    )
    
    print("\nComputed losses:")
    for loss_name, loss_value in losses.items():
        if loss_value is not None:
            print(f"  {loss_name}: {loss_value.item():.4f}")
    
    print(f"\nTotal loss: {losses['total'].item():.4f}")
    print("\n✅ ICN Loss computation implemented successfully!")