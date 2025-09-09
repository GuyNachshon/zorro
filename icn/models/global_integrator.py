"""
Global Intent Integrator for ICN.
Aggregates local intents into package-level consensus through iterative convergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import math


@dataclass
class ConvergenceState:
    """State during convergence iterations."""
    global_embedding: torch.Tensor  # [batch_size, embedding_dim]
    global_intent_dist: torch.Tensor  # [batch_size, n_fixed + n_latent]
    attention_weights: torch.Tensor  # [batch_size, max_units]
    iteration: int
    converged: bool
    drift: float


class GlobalIntegratorOutput(NamedTuple):
    """Output from the Global Intent Integrator."""
    final_global_embedding: torch.Tensor
    final_global_intent: torch.Tensor
    convergence_history: List[ConvergenceState]
    converged: bool
    final_iteration: int
    unit_attention_weights: torch.Tensor


class GlobalIntentIntegrator(nn.Module):
    """
    Global Intent Integrator using attention-based aggregation and convergence loop.
    Iteratively refines package-level intent based on local unit intents.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        n_fixed_intents: int = 15,
        n_latent_intents: int = 10,
        hidden_dim: int = 512,
        n_attention_heads: int = 8,
        max_iterations: int = 6,
        convergence_threshold: float = 0.01,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.n_fixed_intents = n_fixed_intents
        self.n_latent_intents = n_latent_intents
        self.n_total_intents = n_fixed_intents + n_latent_intents
        self.hidden_dim = hidden_dim
        self.n_attention_heads = n_attention_heads
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Attention mechanism for aggregating local units
        self.unit_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Global state update mechanism (GRU-based)
        self.global_update_gru = nn.GRU(
            input_size=embedding_dim + self.n_total_intents,  # embedding + intent dist
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout if dropout > 0 else 0,
            batch_first=True
        )
        
        # Projection layers
        self.embedding_projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout)
        )
        
        self.intent_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.n_total_intents)
        )
        
        # Manifest processing (special handling for manifest units)
        self.manifest_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Feedback mechanism (global -> local influence)
        self.feedback_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Initialize global state embedding
        self.global_state_init = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
    def forward(
        self,
        local_embeddings: torch.Tensor,  # [batch_size, max_units, embedding_dim]
        local_intent_dists: torch.Tensor,  # [batch_size, max_units, n_total_intents]
        unit_masks: torch.Tensor,  # [batch_size, max_units] - valid units
        manifest_embeddings: Optional[torch.Tensor] = None,  # [batch_size, embedding_dim]
        return_history: bool = True
    ) -> GlobalIntegratorOutput:
        """
        Forward pass with convergence loop.
        
        Args:
            local_embeddings: Local unit embeddings
            local_intent_dists: Combined local intent distributions (fixed + latent)
            unit_masks: Mask indicating valid units (padding mask)
            manifest_embeddings: Optional manifest embeddings for initialization
            return_history: Whether to return convergence history
            
        Returns:
            GlobalIntegratorOutput with final states and convergence info
        """
        batch_size, max_units, _ = local_embeddings.shape
        device = local_embeddings.device
        
        # Initialize global state
        global_state = self._initialize_global_state(
            local_embeddings, local_intent_dists, unit_masks, manifest_embeddings
        )
        
        convergence_history = []
        converged = False
        
        # Convergence loop
        for iteration in range(self.max_iterations):
            # Update global state based on local units
            new_global_state, attention_weights = self._update_global_state(
                global_state, local_embeddings, local_intent_dists, unit_masks
            )
            
            # Compute drift (change in global embedding)
            drift = torch.norm(
                new_global_state['embedding'] - global_state['embedding'], 
                dim=-1
            ).mean().item()
            
            # Project to get global intent distribution
            global_intent_dist = self._project_to_intent_distribution(new_global_state)
            
            # Store convergence state
            conv_state = ConvergenceState(
                global_embedding=new_global_state['embedding'].clone(),
                global_intent_dist=global_intent_dist.clone(),
                attention_weights=attention_weights.clone(),
                iteration=iteration,
                converged=drift < self.convergence_threshold,
                drift=drift
            )
            
            if return_history:
                convergence_history.append(conv_state)
            
            # Check convergence
            if drift < self.convergence_threshold:
                converged = True
                break
            
            global_state = new_global_state
        
        # Provide feedback to local units (optional for inference)
        final_attention = self._compute_final_attention_weights(
            global_state['embedding'], local_embeddings, unit_masks
        )
        
        return GlobalIntegratorOutput(
            final_global_embedding=global_state['embedding'],
            final_global_intent=global_intent_dist,
            convergence_history=convergence_history,
            converged=converged,
            final_iteration=iteration,
            unit_attention_weights=final_attention
        )
    
    def _initialize_global_state(
        self,
        local_embeddings: torch.Tensor,
        local_intent_dists: torch.Tensor,
        unit_masks: torch.Tensor,
        manifest_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Initialize global state from local units and manifest."""
        batch_size = local_embeddings.shape[0]
        
        # Mean pooling of local embeddings (with masking)
        if unit_masks is not None:
            mask_expanded = unit_masks.unsqueeze(-1).float()
            masked_embeddings = local_embeddings * mask_expanded
            pooled_embedding = masked_embeddings.sum(1) / (mask_expanded.sum(1) + 1e-8)
        else:
            pooled_embedding = local_embeddings.mean(1)
        
        # Incorporate manifest information if available
        if manifest_embeddings is not None:
            manifest_encoded = self.manifest_encoder(manifest_embeddings)
            # Weighted combination of pooled locals and manifest
            pooled_embedding = 0.7 * pooled_embedding + 0.3 * manifest_encoded
        
        # Initialize hidden state for GRU (num_layers=2, so need 2 layers)
        init_hidden = self.global_state_init.expand(2, batch_size, -1).contiguous()
        
        return {
            'embedding': pooled_embedding,
            'hidden': init_hidden
        }
    
    def _update_global_state(
        self,
        current_state: Dict[str, torch.Tensor],
        local_embeddings: torch.Tensor,
        local_intent_dists: torch.Tensor,
        unit_masks: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Update global state using attention and GRU."""
        
        # Apply attention to aggregate local units
        current_embedding = current_state['embedding'].unsqueeze(1)  # Query
        
        attended_embedding, attention_weights = self.unit_attention(
            query=current_embedding,
            key=local_embeddings,
            value=local_embeddings,
            key_padding_mask=~unit_masks if unit_masks is not None else None
        )
        
        attended_embedding = attended_embedding.squeeze(1)  # Remove sequence dimension
        
        # Aggregate intent distributions using attention weights
        if unit_masks is not None:
            attention_weights = attention_weights.masked_fill(~unit_masks.unsqueeze(1), 0.0)
        
        # Weighted average of local intent distributions
        attended_intent = torch.bmm(
            attention_weights, local_intent_dists
        ).squeeze(1)
        
        # Combine embedding and intent for GRU input
        gru_input = torch.cat([attended_embedding, attended_intent], dim=-1)
        gru_input = gru_input.unsqueeze(1)  # Add sequence dimension
        
        # Update with GRU
        updated_features, new_hidden = self.global_update_gru(
            gru_input, current_state['hidden']
        )
        updated_features = updated_features.squeeze(1)  # Remove sequence dimension
        
        # Project back to embedding space
        new_embedding = self.embedding_projection(updated_features)
        
        return {
            'embedding': new_embedding,
            'hidden': new_hidden
        }, attention_weights.squeeze(1)
    
    def _project_to_intent_distribution(
        self, 
        global_state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Project global state to intent distribution."""
        # Use the updated features from GRU
        gru_output = global_state['hidden'][-1]  # Last layer, last time step
        
        # Project to intent space
        intent_logits = self.intent_projection(gru_output)
        intent_dist = F.softmax(intent_logits, dim=-1)
        
        return intent_dist
    
    def _compute_final_attention_weights(
        self,
        global_embedding: torch.Tensor,
        local_embeddings: torch.Tensor,
        unit_masks: torch.Tensor
    ) -> torch.Tensor:
        """Compute final attention weights for interpretability."""
        
        # Simple dot-product attention for final weights
        similarity = torch.bmm(
            global_embedding.unsqueeze(1),  # [batch, 1, dim]
            local_embeddings.transpose(-1, -2)  # [batch, dim, max_units]
        ).squeeze(1)  # [batch, max_units]
        
        # Apply mask and softmax
        if unit_masks is not None:
            similarity = similarity.masked_fill(~unit_masks, -1e9)
        
        attention_weights = F.softmax(similarity, dim=-1)
        
        return attention_weights
    
    def get_convergence_metrics(
        self, 
        convergence_history: List[ConvergenceState]
    ) -> Dict[str, float]:
        """Extract convergence metrics for analysis."""
        if not convergence_history:
            return {}
        
        drifts = [state.drift for state in convergence_history]
        iterations_to_converge = len(convergence_history)
        
        # Find when convergence was achieved
        for i, state in enumerate(convergence_history):
            if state.converged:
                iterations_to_converge = i + 1
                break
        
        return {
            'iterations_to_converge': iterations_to_converge,
            'final_drift': drifts[-1],
            'max_drift': max(drifts),
            'mean_drift': sum(drifts) / len(drifts),
            'converged': convergence_history[-1].converged
        }
    
    def compute_divergence_metrics(
        self,
        local_intent_dists: torch.Tensor,
        global_intent_dist: torch.Tensor,
        unit_masks: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute divergence metrics for dual detection channels.
        
        Returns:
            Dictionary with mean/max divergence and per-unit divergences
        """
        batch_size, max_units, _ = local_intent_dists.shape
        
        # Expand global distribution for broadcasting
        global_expanded = global_intent_dist.unsqueeze(1).expand(-1, max_units, -1)
        
        # Compute KL divergence for each unit
        eps = 1e-8
        local_log = torch.log(local_intent_dists + eps)
        
        # KL(local || global) for each unit
        kl_divergences = F.kl_div(
            local_log, 
            global_expanded, 
            reduction='none'
        ).sum(dim=-1)  # [batch_size, max_units]
        
        # Apply mask
        if unit_masks is not None:
            kl_divergences = kl_divergences * unit_masks.float()
        
        # Compute statistics
        valid_counts = unit_masks.sum(dim=-1).float() if unit_masks is not None else torch.full((batch_size,), max_units, device=kl_divergences.device)
        
        mean_divergence = kl_divergences.sum(dim=-1) / (valid_counts + eps)
        max_divergence = kl_divergences.max(dim=-1)[0]
        
        return {
            'mean_divergence': mean_divergence,
            'max_divergence': max_divergence,
            'per_unit_divergence': kl_divergences
        }


class ManifestProcessor(nn.Module):
    """Special processor for manifest files (package.json, setup.py, etc.)"""
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        
        self.processor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Special attention for manifest features
        self.feature_weights = nn.Parameter(torch.ones(5))  # dependencies, scripts, etc.
        
    def forward(self, manifest_embedding: torch.Tensor) -> torch.Tensor:
        """Process manifest embedding with special attention."""
        processed = self.processor(manifest_embedding)
        
        # Could add special logic for extracting structured manifest info
        # For now, just return processed embedding
        return processed


if __name__ == "__main__":
    # Test the Global Intent Integrator
    print("Testing Global Intent Integrator...")
    
    batch_size = 4
    max_units = 8
    embedding_dim = 256
    n_fixed_intents = 15
    n_latent_intents = 10
    n_total_intents = n_fixed_intents + n_latent_intents
    
    # Create model
    integrator = GlobalIntentIntegrator(
        embedding_dim=embedding_dim,
        n_fixed_intents=n_fixed_intents,
        n_latent_intents=n_latent_intents,
        hidden_dim=128,
        max_iterations=4
    )
    
    # Create dummy inputs with variable number of units per package
    local_embeddings = torch.randn(batch_size, max_units, embedding_dim)
    local_intent_dists = F.softmax(torch.randn(batch_size, max_units, n_total_intents), dim=-1)
    
    # Create masks for variable lengths
    unit_counts = [3, 5, 2, 6]  # Number of units per package
    unit_masks = torch.zeros(batch_size, max_units, dtype=torch.bool)
    for i, count in enumerate(unit_counts):
        unit_masks[i, :count] = True
    
    manifest_embeddings = torch.randn(batch_size, embedding_dim)
    
    # Forward pass
    output = integrator(
        local_embeddings=local_embeddings,
        local_intent_dists=local_intent_dists,
        unit_masks=unit_masks,
        manifest_embeddings=manifest_embeddings,
        return_history=True
    )
    
    print(f"Final global embedding shape: {output.final_global_embedding.shape}")
    print(f"Final global intent shape: {output.final_global_intent.shape}")
    print(f"Converged: {output.converged}")
    print(f"Final iteration: {output.final_iteration}")
    print(f"Convergence history length: {len(output.convergence_history)}")
    
    # Test convergence metrics
    conv_metrics = integrator.get_convergence_metrics(output.convergence_history)
    print(f"\nConvergence metrics:")
    for key, value in conv_metrics.items():
        print(f"  {key}: {value}")
    
    # Test divergence computation
    div_metrics = integrator.compute_divergence_metrics(
        local_intent_dists, output.final_global_intent, unit_masks
    )
    print(f"\nDivergence metrics:")
    print(f"  Mean divergence: {div_metrics['mean_divergence'].mean().item():.4f}")
    print(f"  Max divergence: {div_metrics['max_divergence'].mean().item():.4f}")
    
    # Show convergence history
    print(f"\nConvergence drift per iteration:")
    for i, state in enumerate(output.convergence_history):
        print(f"  Iteration {i}: drift={state.drift:.6f}, converged={state.converged}")
    
    print("\n✅ Global Intent Integrator implemented successfully!")