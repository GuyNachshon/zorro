"""
AMIL (Attention-based Multiple Instance Learning) Model.
Core architecture with attention pooling and package-level classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math
import logging

from .config import AMILConfig
from .feature_extractor import UnitFeatures

logger = logging.getLogger(__name__)


@dataclass
class AMILOutput:
    """Output from AMIL model with interpretability information."""

    # Core predictions
    package_logits: torch.Tensor  # Raw logits
    package_probability: torch.Tensor  # Sigmoid probabilities
    package_prediction: torch.Tensor  # Binary prediction (0/1)

    # Attention information for interpretability
    attention_weights: torch.Tensor  # Per-unit attention weights
    attention_scores: torch.Tensor  # Raw attention scores before softmax

    # Top-K most attended units
    top_k_indices: torch.Tensor  # Indices of top-K units
    top_k_weights: torch.Tensor  # Attention weights of top-K units

    # Package-level embedding
    package_embedding: torch.Tensor  # Weighted sum of unit embeddings

    # Optional fields with defaults (must come last)
    top_k_unit_names: Optional[List[str]] = None  # Names of top-K units (if available)
    num_units: int = 0
    processing_time: Optional[float] = None


class MultiHeadAttentionMIL(nn.Module):
    """Multi-head attention mechanism for MIL pooling."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Attention projections
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
        # Learnable query for MIL (represents "what makes a unit suspicious")
        self.mil_query = nn.Parameter(torch.randn(embed_dim))
        
    def forward(self, unit_embeddings: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with MIL attention.
        
        Args:
            unit_embeddings: (num_units, embed_dim) tensor of unit embeddings
            mask: (num_units,) boolean mask for valid units
            
        Returns:
            weighted_embedding: (embed_dim,) package-level embedding
            attention_weights: (num_units,) attention weights
        """
        num_units, embed_dim = unit_embeddings.shape
        
        # Expand MIL query for multi-head attention
        mil_query = self.mil_query.unsqueeze(0).expand(num_units, -1)  # (num_units, embed_dim)
        
        # Multi-head projections
        Q = self.query_proj(mil_query)  # (num_units, embed_dim)
        K = self.key_proj(unit_embeddings)  # (num_units, embed_dim)
        V = self.value_proj(unit_embeddings)  # (num_units, embed_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(num_units, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, num_units, head_dim)
        K = K.view(num_units, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, num_units, head_dim)
        V = V.view(num_units, self.num_heads, self.head_dim).transpose(0, 1)  # (num_heads, num_units, head_dim)
        
        # Attention computation: Q @ K^T / sqrt(d_k)
        attention_scores = torch.bmm(Q, K.transpose(-2, -1)) / self.scale  # (num_heads, num_units, num_units)
        
        # For MIL, we want attention from query to all units (not unit-to-unit)
        # Take diagonal to get query-to-unit attention
        attention_scores = torch.diagonal(attention_scores, dim1=-2, dim2=-1)  # (num_heads, num_units)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask.unsqueeze(0), float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (num_heads, num_units)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum of values
        weighted_values = torch.bmm(attention_weights.unsqueeze(-2), V.transpose(0, 1))  # (num_heads, 1, head_dim)
        weighted_values = weighted_values.squeeze(-2).transpose(0, 1).contiguous()  # (1, num_heads * head_dim)
        weighted_values = weighted_values.view(embed_dim)  # (embed_dim,)
        
        # Output projection
        package_embedding = self.out_proj(weighted_values)
        
        # Average attention weights across heads for interpretability
        final_attention_weights = attention_weights.mean(dim=0)  # (num_units,)
        
        return package_embedding, final_attention_weights


class AMILModel(nn.Module):
    """
    Complete AMIL model for malicious package detection.
    
    Architecture:
    1. Feature extraction and fusion (handled by AMILFeatureExtractor)
    2. Multi-head attention pooling (MultiHeadAttentionMIL)
    3. Package-level classifier (2-layer MLP)
    """
    
    def __init__(self, config: AMILConfig):
        super().__init__()
        self.config = config
        
        # Attention-based MIL pooling
        self.attention_pooling = MultiHeadAttentionMIL(
            embed_dim=config.unit_embedding_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout
        )
        
        # Package-level classifier
        self.classifier = nn.Sequential(
            # First layer
            nn.Linear(config.unit_embedding_dim, config.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            
            # Second layer
            nn.Linear(config.classifier_hidden_dim, config.classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout / 2),
            
            # Output layer
            nn.Linear(config.classifier_hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Parameter):
                nn.init.normal_(module, std=0.02)
    
    def forward(self, unit_embeddings: torch.Tensor, 
                unit_names: Optional[List[str]] = None,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = True) -> AMILOutput:
        """
        Forward pass through AMIL model.
        
        Args:
            unit_embeddings: (num_units, unit_embedding_dim) tensor
            unit_names: Optional list of unit names for interpretability
            mask: Optional (num_units,) boolean mask for valid units
            return_attention: Whether to compute attention information
            
        Returns:
            AMILOutput with predictions and attention information
        """
        num_units = unit_embeddings.shape[0]
        
        # Handle empty packages (shouldn't happen in practice)
        if num_units == 0:
            return self._empty_package_output()
        
        # Apply mask if provided
        if mask is not None:
            unit_embeddings = unit_embeddings * mask.unsqueeze(-1)
        
        # Attention-based pooling
        package_embedding, attention_weights = self.attention_pooling(unit_embeddings, mask)
        
        # Package classification
        logits = self.classifier(package_embedding)  # (1,)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).float()
        
        # Prepare output
        output = AMILOutput(
            package_logits=logits,
            package_probability=probabilities,
            package_prediction=predictions,
            attention_weights=attention_weights,
            attention_scores=attention_weights,  # Same as weights after softmax
            package_embedding=package_embedding,
            num_units=num_units
        )
        
        # Add top-K attention analysis
        if return_attention:
            self._add_top_k_analysis(output, unit_names)
        
        return output
    
    def _add_top_k_analysis(self, output: AMILOutput, unit_names: Optional[List[str]] = None):
        """Add top-K attention analysis to output."""
        k = min(self.config.max_units_per_package // 10, output.num_units, 10)  # Top-10 or fewer
        
        # Get top-K indices and weights
        top_k_weights, top_k_indices = torch.topk(output.attention_weights, k)
        
        output.top_k_indices = top_k_indices
        output.top_k_weights = top_k_weights
        
        # Add unit names if provided
        if unit_names is not None:
            output.top_k_unit_names = [unit_names[i] for i in top_k_indices.cpu().tolist()]
    
    def _empty_package_output(self) -> AMILOutput:
        """Handle edge case of empty package."""
        return AMILOutput(
            package_logits=torch.tensor([0.0]),
            package_probability=torch.tensor([0.5]),
            package_prediction=torch.tensor([0.0]),
            attention_weights=torch.empty(0),
            attention_scores=torch.empty(0),
            top_k_indices=torch.empty(0, dtype=torch.long),
            top_k_weights=torch.empty(0),
            package_embedding=torch.zeros(self.config.unit_embedding_dim),
            num_units=0
        )
    
    def predict_package(self, unit_embeddings: torch.Tensor, 
                       unit_names: Optional[List[str]] = None,
                       threshold: float = 0.5) -> Dict[str, Any]:
        """
        High-level prediction interface for a single package.
        
        Args:
            unit_embeddings: (num_units, unit_embedding_dim) tensor
            unit_names: Optional list of unit names
            threshold: Classification threshold (default 0.5)
            
        Returns:
            Dictionary with prediction results and explanations
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(unit_embeddings, unit_names, return_attention=True)
        
        # Format results
        result = {
            "is_malicious": bool(output.package_probability.item() > threshold),
            "malicious_probability": float(output.package_probability.item()),
            "confidence": float(abs(output.package_probability.item() - 0.5) * 2),  # Distance from 0.5
            
            # Top suspicious units
            "top_suspicious_units": [],
            "num_units_analyzed": output.num_units,
            
            # Raw outputs for advanced users
            "raw_logits": float(output.package_logits.item()),
            "attention_entropy": float(self._calculate_attention_entropy(output.attention_weights)),
        }
        
        # Add top-K unit analysis
        if output.top_k_unit_names is not None:
            for i, (name, weight) in enumerate(zip(output.top_k_unit_names, output.top_k_weights)):
                result["top_suspicious_units"].append({
                    "unit_name": name,
                    "attention_weight": float(weight.item()),
                    "rank": i + 1
                })
        
        return result
    
    def _calculate_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Calculate entropy of attention distribution (higher = more distributed)."""
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-8
        attention_weights = attention_weights + epsilon
        entropy = -torch.sum(attention_weights * torch.log(attention_weights))
        return entropy
    
    def get_attention_explanation(self, unit_embeddings: torch.Tensor, 
                                 unit_names: List[str],
                                 unit_features: Optional[List[UnitFeatures]] = None) -> Dict[str, Any]:
        """
        Generate detailed explanation of model attention.
        
        Args:
            unit_embeddings: (num_units, unit_embedding_dim) tensor
            unit_names: List of unit names
            unit_features: Optional list of UnitFeatures for API analysis
            
        Returns:
            Detailed explanation dictionary
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(unit_embeddings, unit_names, return_attention=True)
        
        explanation = {
            "package_verdict": {
                "is_malicious": bool(output.package_probability.item() > 0.5),
                "probability": float(output.package_probability.item()),
                "confidence": float(abs(output.package_probability.item() - 0.5) * 2)
            },
            
            "attention_analysis": {
                "entropy": float(self._calculate_attention_entropy(output.attention_weights)),
                "max_attention": float(output.attention_weights.max().item()),
                "min_attention": float(output.attention_weights.min().item()),
                "attention_std": float(output.attention_weights.std().item())
            },
            
            "unit_rankings": []
        }
        
        # Rank all units by attention
        sorted_indices = torch.argsort(output.attention_weights, descending=True)
        
        for rank, idx in enumerate(sorted_indices[:10]):  # Top 10
            unit_info = {
                "rank": rank + 1,
                "unit_name": unit_names[idx],
                "attention_weight": float(output.attention_weights[idx].item()),
                "attention_percentage": float(output.attention_weights[idx].item() * 100)
            }
            
            # Add API analysis if features available
            if unit_features and idx < len(unit_features):
                features = unit_features[idx]
                unit_info["api_analysis"] = {
                    "suspicious_apis": [
                        api for api, count in features.api_counts.items() 
                        if count > 0 and any(suspicious in api for suspicious in 
                                           ["subprocess", "eval", "obfuscation", "env"])
                    ],
                    "total_api_calls": sum(features.api_counts.values()),
                    "entropy_score": features.shannon_entropy,
                    "obfuscation_score": features.obfuscation_score
                }
            
            explanation["unit_rankings"].append(unit_info)
        
        return explanation


def create_amil_model(config: AMILConfig, device: Optional[torch.device] = None) -> AMILModel:
    """Create and initialize AMIL model."""
    model = AMILModel(config)
    
    if device is not None:
        model = model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Created AMIL model with {total_params:,} total parameters")
    logger.info(f"  Trainable: {trainable_params:,}")
    logger.info(f"  Unit embedding dim: {config.unit_embedding_dim}")
    logger.info(f"  Attention heads: {config.attention_heads}")
    
    return model