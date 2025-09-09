"""
Pooling strategies for aggregating unit-level embeddings to package-level representation.
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .config import NeoBERTConfig

logger = logging.getLogger(__name__)


@dataclass
class PoolingOutput:
    """Output from pooling layer with interpretability information."""
    
    package_embedding: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    unit_scores: Optional[torch.Tensor] = None
    suspicious_unit_indices: Optional[List[int]] = None
    pooling_metadata: Optional[Dict[str, Any]] = None


class BasePooling(nn.Module, ABC):
    """Abstract base class for pooling strategies."""
    
    def __init__(self, config: NeoBERTConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.projection_dim
    
    @abstractmethod
    def forward(self, unit_embeddings: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> PoolingOutput:
        """Forward pass for pooling."""
        pass
    
    def get_output_dim(self) -> int:
        """Get output dimension of pooled embeddings."""
        return self.input_dim


class MeanPooling(BasePooling):
    """Simple mean pooling baseline."""
    
    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)
        
        # Optional learnable scaling
        self.learnable_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, unit_embeddings: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> PoolingOutput:
        """
        Args:
            unit_embeddings: (batch_size, num_units, embedding_dim) or (num_units, embedding_dim)
            mask: (batch_size, num_units) or (num_units,) - 1 for valid units, 0 for padding
        """
        
        if unit_embeddings.dim() == 2:
            # Single package case
            unit_embeddings = unit_embeddings.unsqueeze(0)
            single_package = True
        else:
            single_package = False
        
        batch_size, num_units, embedding_dim = unit_embeddings.shape
        
        if mask is None:
            # No mask - simple mean
            package_embedding = unit_embeddings.mean(dim=1)
        else:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            
            # Masked mean
            mask_expanded = mask.unsqueeze(-1).expand_as(unit_embeddings)
            masked_embeddings = unit_embeddings * mask_expanded
            
            # Sum and normalize by valid units
            sum_embeddings = masked_embeddings.sum(dim=1)
            valid_counts = mask.sum(dim=1, keepdim=True).float()
            valid_counts = torch.clamp(valid_counts, min=1)  # Avoid division by zero
            
            package_embedding = sum_embeddings / valid_counts
        
        # Apply learnable scaling
        package_embedding = package_embedding * self.learnable_scale
        
        if single_package:
            package_embedding = package_embedding.squeeze(0)
        
        return PoolingOutput(
            package_embedding=package_embedding,
            pooling_metadata={"pooling_type": "mean"}
        )


class AttentionPooling(BasePooling):
    """Learned attention pooling with interpretability."""
    
    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)
        
        # Multi-head attention parameters
        self.num_heads = config.attention_heads
        self.head_dim = self.input_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.input_dim, "input_dim must be divisible by num_heads"
        
        # Attention computation
        self.query_projection = nn.Linear(self.input_dim, self.input_dim)
        self.key_projection = nn.Linear(self.input_dim, self.input_dim)
        self.value_projection = nn.Linear(self.input_dim, self.input_dim)
        
        # Output projection
        self.output_projection = nn.Linear(self.input_dim, self.input_dim)
        
        # Context vector for attention (learnable query)
        self.context_vector = nn.Parameter(torch.randn(self.input_dim))
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.input_dim)
        
        # Temperature for attention sharpening
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, unit_embeddings: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> PoolingOutput:
        """
        Multi-head attention pooling with interpretability.
        """
        
        if unit_embeddings.dim() == 2:
            unit_embeddings = unit_embeddings.unsqueeze(0)
            single_package = True
        else:
            single_package = False
        
        batch_size, num_units, embedding_dim = unit_embeddings.shape
        
        # Multi-head attention computation
        Q = self.query_projection(unit_embeddings)  # (batch_size, num_units, input_dim)
        K = self.key_projection(unit_embeddings)
        V = self.value_projection(unit_embeddings)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_units, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_units, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_units, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores using context vector as query
        context_expanded = self.context_vector.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        context_expanded = context_expanded.expand(batch_size, self.num_heads, 1, self.head_dim)
        
        # Attention scores: context_vector @ K^T
        attention_scores = torch.matmul(context_expanded, K.transpose(-2, -1))  # (batch_size, num_heads, 1, num_units)
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply temperature scaling
        attention_scores = attention_scores / torch.clamp(self.temperature, min=0.1)
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, num_units)
            attention_scores = attention_scores.masked_fill(mask_expanded == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, 1, num_units)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)  # (batch_size, num_heads, 1, head_dim)
        
        # Concatenate heads and reshape
        attended_values = attended_values.transpose(1, 2).contiguous()  # (batch_size, 1, num_heads, head_dim)
        package_embedding = attended_values.view(batch_size, self.input_dim)  # (batch_size, input_dim)
        
        # Output projection and normalization
        package_embedding = self.output_projection(package_embedding)
        package_embedding = self.layer_norm(package_embedding)
        
        # Average attention weights across heads for interpretability
        mean_attention = attention_weights.mean(dim=1).squeeze(2)  # (batch_size, num_units)
        
        # Find most attended units
        top_k = min(5, num_units)
        suspicious_indices = []
        
        for batch_idx in range(batch_size):
            if mask is not None:
                valid_units = mask[batch_idx].nonzero(as_tuple=True)[0]
                if len(valid_units) == 0:
                    continue
                valid_attention = mean_attention[batch_idx][valid_units]
                top_indices_local = torch.topk(valid_attention, min(top_k, len(valid_units))).indices
                top_indices_global = valid_units[top_indices_local].tolist()
            else:
                top_indices_global = torch.topk(mean_attention[batch_idx], top_k).indices.tolist()
            
            suspicious_indices.append(top_indices_global)
        
        if single_package:
            package_embedding = package_embedding.squeeze(0)
            mean_attention = mean_attention.squeeze(0)
            suspicious_indices = suspicious_indices[0] if suspicious_indices else []
        
        return PoolingOutput(
            package_embedding=package_embedding,
            attention_weights=mean_attention,
            suspicious_unit_indices=suspicious_indices,
            pooling_metadata={
                "pooling_type": "attention",
                "num_heads": self.num_heads,
                "temperature": self.temperature.item(),
                "attention_entropy": self._compute_attention_entropy(mean_attention)
            }
        )
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention distribution."""
        
        if attention_weights.dim() > 1:
            # Multiple packages - compute mean entropy
            entropies = []
            for i in range(attention_weights.size(0)):
                entropy = self._single_entropy(attention_weights[i])
                entropies.append(entropy)
            return float(torch.tensor(entropies).mean())
        else:
            return self._single_entropy(attention_weights)
    
    def _single_entropy(self, weights: torch.Tensor) -> float:
        """Compute entropy for single attention distribution."""
        
        # Add small epsilon to avoid log(0)
        weights = weights + 1e-8
        entropy = -(weights * torch.log(weights)).sum()
        
        # Normalize by max possible entropy
        max_entropy = math.log(weights.size(0))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0


class MILPooling(BasePooling):
    """Multiple Instance Learning pooling with instance-level scoring."""
    
    def __init__(self, config: NeoBERTConfig):
        super().__init__(config)
        
        # MIL attention network
        self.attention_hidden_dim = config.mil_attention_dim
        
        self.attention_network = nn.Sequential(
            nn.Linear(self.input_dim, self.attention_hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(self.attention_hidden_dim, 1)
        )
        
        # Instance classifier (for interpretability)
        self.instance_classifier = nn.Sequential(
            nn.Linear(self.input_dim, config.mil_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.mil_hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Gating mechanism
        self.gate_network = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 4),
            nn.ReLU(),
            nn.Linear(self.input_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, unit_embeddings: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> PoolingOutput:
        """
        MIL pooling with attention mechanism and instance scoring.
        """
        
        if unit_embeddings.dim() == 2:
            unit_embeddings = unit_embeddings.unsqueeze(0)
            single_package = True
        else:
            single_package = False
        
        batch_size, num_units, embedding_dim = unit_embeddings.shape
        
        # Compute instance scores (probability each unit is malicious)
        unit_scores = self.instance_classifier(unit_embeddings)  # (batch_size, num_units, 1)
        unit_scores = unit_scores.squeeze(-1)  # (batch_size, num_units)
        
        # Compute attention weights
        attention_logits = self.attention_network(unit_embeddings).squeeze(-1)  # (batch_size, num_units)
        
        # Apply mask to attention if provided
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            attention_logits = attention_logits.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(attention_logits, dim=-1)  # (batch_size, num_units)
        
        # Compute gating scores (how much each unit should contribute)
        gate_scores = self.gate_network(unit_embeddings).squeeze(-1)  # (batch_size, num_units)
        
        # Combine attention and gating
        combined_weights = attention_weights * gate_scores
        
        # Renormalize
        if mask is not None:
            combined_weights = combined_weights * mask.float()
        
        weight_sums = combined_weights.sum(dim=-1, keepdim=True)
        weight_sums = torch.clamp(weight_sums, min=1e-8)
        combined_weights = combined_weights / weight_sums
        
        # Weighted aggregation
        package_embedding = torch.sum(
            combined_weights.unsqueeze(-1) * unit_embeddings, 
            dim=1
        )  # (batch_size, embedding_dim)
        
        # Find most suspicious units based on combined score
        suspicion_scores = unit_scores * attention_weights
        
        top_k = min(5, num_units)
        suspicious_indices = []
        
        for batch_idx in range(batch_size):
            if mask is not None:
                valid_units = mask[batch_idx].nonzero(as_tuple=True)[0]
                if len(valid_units) == 0:
                    continue
                valid_suspicion = suspicion_scores[batch_idx][valid_units]
                top_indices_local = torch.topk(valid_suspicion, min(top_k, len(valid_units))).indices
                top_indices_global = valid_units[top_indices_local].tolist()
            else:
                top_indices_global = torch.topk(suspicion_scores[batch_idx], top_k).indices.tolist()
            
            suspicious_indices.append(top_indices_global)
        
        if single_package:
            package_embedding = package_embedding.squeeze(0)
            attention_weights = attention_weights.squeeze(0)
            unit_scores = unit_scores.squeeze(0)
            suspicious_indices = suspicious_indices[0] if suspicious_indices else []
        
        return PoolingOutput(
            package_embedding=package_embedding,
            attention_weights=attention_weights,
            unit_scores=unit_scores,
            suspicious_unit_indices=suspicious_indices,
            pooling_metadata={
                "pooling_type": "mil",
                "max_instance_score": float(unit_scores.max()) if unit_scores.numel() > 0 else 0.0,
                "mean_instance_score": float(unit_scores.mean()) if unit_scores.numel() > 0 else 0.0,
                "attention_entropy": self._compute_attention_entropy(attention_weights),
                "high_suspicion_units": int((unit_scores > 0.7).sum()) if unit_scores.numel() > 0 else 0
            }
        )
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention distribution (same as AttentionPooling)."""
        
        if attention_weights.dim() > 1:
            entropies = []
            for i in range(attention_weights.size(0)):
                entropy = self._single_entropy(attention_weights[i])
                entropies.append(entropy)
            return float(torch.tensor(entropies).mean())
        else:
            return self._single_entropy(attention_weights)
    
    def _single_entropy(self, weights: torch.Tensor) -> float:
        """Compute entropy for single attention distribution."""
        
        weights = weights + 1e-8
        entropy = -(weights * torch.log(weights)).sum()
        max_entropy = math.log(weights.size(0))
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0
    
    def get_instance_predictions(self, unit_embeddings: torch.Tensor, 
                               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get instance-level predictions for analysis."""
        
        with torch.no_grad():
            if unit_embeddings.dim() == 2:
                unit_embeddings = unit_embeddings.unsqueeze(0)
            
            instance_scores = self.instance_classifier(unit_embeddings).squeeze(-1)
            
            if mask is not None:
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0)
                instance_scores = instance_scores * mask.float()
            
            return instance_scores


def create_pooling_layer(config: NeoBERTConfig) -> BasePooling:
    """Create appropriate pooling layer based on configuration."""
    
    if config.pooling_strategy == "mean":
        return MeanPooling(config)
    elif config.pooling_strategy == "attention":
        return AttentionPooling(config)
    elif config.pooling_strategy == "mil":
        return MILPooling(config)
    else:
        logger.warning(f"Unknown pooling strategy: {config.pooling_strategy}. Using mean pooling.")
        return MeanPooling(config)