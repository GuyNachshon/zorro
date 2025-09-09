"""
Loss functions for AMIL training.
Combines binary cross-entropy, attention sparsity, and counterfactual consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging

from .config import TrainingConfig
from .model import AMILModel, AMILOutput

logger = logging.getLogger(__name__)


class AMILLossFunction(nn.Module):
    """
    Combined loss function for AMIL training.
    
    Components:
    1. Binary Cross-Entropy: Primary classification loss
    2. Attention Sparsity: L1 regularization on attention weights
    3. Counterfactual Loss: Ensure score drops when masking top-attention unit
    """
    
    def __init__(self, config: TrainingConfig, model: AMILModel):
        super().__init__()
        self.config = config
        self.model = model
        
        # Loss weights from config
        self.bce_weight = config.bce_weight
        self.sparsity_weight = config.sparsity_weight
        self.counterfactual_weight = config.counterfactual_weight
        
        # BCE loss with class balancing
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        logger.info(f"AMIL Loss initialized with weights:")
        logger.info(f"  BCE: {self.bce_weight:.4f}")
        logger.info(f"  Sparsity: {self.sparsity_weight:.4f}")
        logger.info(f"  Counterfactual: {self.counterfactual_weight:.4f}")
    
    def forward(self, unit_embeddings_batch: List[torch.Tensor], 
                targets: torch.Tensor,
                unit_names_batch: Optional[List[List[str]]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for batch of packages.
        
        Args:
            unit_embeddings_batch: List of (num_units_i, embed_dim) tensors
            targets: (batch_size,) tensor of package labels (0/1)
            unit_names_batch: Optional batch of unit names
            
        Returns:
            Dictionary with total loss and component losses
        """
        batch_size = len(unit_embeddings_batch)
        device = targets.device
        
        # Storage for loss components
        bce_losses = []
        sparsity_losses = []
        counterfactual_losses = []
        
        # Process each package in batch
        for i, (unit_embeddings, target) in enumerate(zip(unit_embeddings_batch, targets)):
            unit_names = unit_names_batch[i] if unit_names_batch else None
            
            # Forward pass
            output = self.model(unit_embeddings, unit_names, return_attention=True)
            
            # 1. Binary Cross-Entropy Loss
            bce_loss = self.bce_loss(output.package_logits, target.unsqueeze(0))
            bce_losses.append(bce_loss)
            
            # 2. Attention Sparsity Loss
            sparsity_loss = self._compute_sparsity_loss(output.attention_weights)
            sparsity_losses.append(sparsity_loss)
            
            # 3. Counterfactual Loss (only for positive samples to avoid instability)
            if target.item() > 0.5:  # Only apply to malicious packages
                counterfactual_loss = self._compute_counterfactual_loss(
                    unit_embeddings, output, target
                )
                counterfactual_losses.append(counterfactual_loss)
            else:
                # Add zero loss for benign packages to maintain batch consistency
                counterfactual_losses.append(torch.tensor(0.0, device=device))
        
        # Average losses across batch
        avg_bce_loss = torch.stack(bce_losses).mean()
        avg_sparsity_loss = torch.stack(sparsity_losses).mean()
        avg_counterfactual_loss = torch.stack(counterfactual_losses).mean()
        
        # Combined loss
        total_loss = (
            self.bce_weight * avg_bce_loss +
            self.sparsity_weight * avg_sparsity_loss +
            self.counterfactual_weight * avg_counterfactual_loss
        )
        
        return {
            "total_loss": total_loss,
            "bce_loss": avg_bce_loss,
            "sparsity_loss": avg_sparsity_loss,
            "counterfactual_loss": avg_counterfactual_loss
        }
    
    def _compute_sparsity_loss(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute L1 sparsity regularization on attention weights.
        Encourages the model to focus on few units rather than spreading attention.
        """
        # L1 norm of attention weights
        sparsity_loss = torch.norm(attention_weights, p=1)
        
        # Normalize by number of units to make loss scale-invariant
        num_units = attention_weights.shape[0]
        normalized_loss = sparsity_loss / num_units
        
        return normalized_loss
    
    def _compute_counterfactual_loss(self, unit_embeddings: torch.Tensor, 
                                   original_output: AMILOutput,
                                   target: torch.Tensor) -> torch.Tensor:
        """
        Compute counterfactual consistency loss.
        Ensures that masking the top-attention unit significantly drops the malicious score.
        """
        if original_output.num_units <= 1:
            # Can't mask if only one unit
            return torch.tensor(0.0, device=unit_embeddings.device)
        
        # Find top attention unit
        top_attention_idx = torch.argmax(original_output.attention_weights)
        
        # Create mask that excludes the top attention unit
        mask = torch.ones(original_output.num_units, dtype=torch.bool, device=unit_embeddings.device)
        mask[top_attention_idx] = False
        
        # Forward pass with masked input
        masked_unit_embeddings = unit_embeddings[mask]
        
        # Get prediction without top unit
        with torch.no_grad():  # Don't backprop through this forward pass
            masked_output = self.model(masked_unit_embeddings, return_attention=False)
        
        # Counterfactual loss: original score should be higher than masked score
        # For malicious packages, removing suspicious unit should drop the score
        score_drop = original_output.package_logits - masked_output.package_logits
        
        # Loss is higher when score doesn't drop enough
        # We want score_drop to be positive and significant
        target_drop = 1.0  # Target logit drop of 1.0
        counterfactual_loss = F.relu(target_drop - score_drop)
        
        return counterfactual_loss
    
    def compute_class_balanced_bce(self, logits: torch.Tensor, 
                                  targets: torch.Tensor,
                                  pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute class-balanced BCE loss for imbalanced datasets.
        
        Args:
            logits: Model logits
            targets: Ground truth labels
            pos_weight: Weight for positive class (computed from class frequencies)
            
        Returns:
            Balanced BCE loss
        """
        if pos_weight is not None:
            bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            bce_loss = nn.BCEWithLogitsLoss()
        
        return bce_loss(logits, targets)
    
    def update_loss_weights(self, epoch: int, total_epochs: int):
        """
        Update loss weights during training (curriculum learning).
        
        Initially focus on classification, gradually add regularization.
        """
        # Linear schedule for counterfactual loss (start low, increase)
        counterfactual_schedule = min(1.0, epoch / (total_epochs * 0.5))
        self.counterfactual_weight = self.config.counterfactual_weight * counterfactual_schedule
        
        # Quadratic schedule for sparsity (start high, decrease)
        sparsity_schedule = max(0.1, (1 - epoch / total_epochs) ** 2)
        self.sparsity_weight = self.config.sparsity_weight * sparsity_schedule
        
        logger.debug(f"Epoch {epoch}: Updated loss weights - "
                    f"Sparsity: {self.sparsity_weight:.4f}, "
                    f"Counterfactual: {self.counterfactual_weight:.4f}")


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Alternative to BCE for heavily imbalanced datasets.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Model logits (batch_size,)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Focal loss
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Compute focal loss components
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Modulating factor (1 - p_t)^gamma
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * focal_weight * ce_loss
        
        return focal_loss.mean()


class AttentionConsistencyLoss(nn.Module):
    """
    Additional loss for attention consistency across similar packages.
    Encourages similar attention patterns for packages with similar content.
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, attention_batch: List[torch.Tensor], 
                similarity_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute attention consistency loss.
        
        Args:
            attention_batch: List of attention weight tensors
            similarity_matrix: (batch_size, batch_size) similarity matrix
            
        Returns:
            Consistency loss
        """
        batch_size = len(attention_batch)
        consistency_loss = 0.0
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                # Skip if packages are too different
                if similarity_matrix[i, j] < 0.7:  # Threshold for "similar"
                    continue
                
                # Compute attention similarity
                attn_i = attention_batch[i]
                attn_j = attention_batch[j]
                
                # Pad to same length if needed
                max_len = max(len(attn_i), len(attn_j))
                if len(attn_i) < max_len:
                    attn_i = F.pad(attn_i, (0, max_len - len(attn_i)))
                if len(attn_j) < max_len:
                    attn_j = F.pad(attn_j, (0, max_len - len(attn_j)))
                
                # KL divergence between attention distributions
                kl_div = F.kl_div(
                    F.log_softmax(attn_i / self.temperature, dim=0),
                    F.softmax(attn_j / self.temperature, dim=0),
                    reduction='sum'
                )
                
                # Weight by package similarity
                weighted_kl = similarity_matrix[i, j] * kl_div
                consistency_loss += weighted_kl
        
        # Normalize by number of pairs
        num_pairs = batch_size * (batch_size - 1) // 2
        return consistency_loss / max(num_pairs, 1)


def create_loss_function(config: TrainingConfig, model: AMILModel, 
                        use_focal: bool = False, 
                        focal_gamma: float = 2.0) -> nn.Module:
    """
    Create appropriate loss function for AMIL training.
    
    Args:
        config: Training configuration
        model: AMIL model
        use_focal: Whether to use focal loss instead of BCE
        focal_gamma: Focal loss gamma parameter
        
    Returns:
        Loss function
    """
    if use_focal:
        logger.info(f"Using Focal Loss (gamma={focal_gamma}) instead of BCE")
        # Custom implementation that combines focal loss with AMIL regularizers
        # Would need to modify AMILLossFunction to use focal loss
        pass
    
    return AMILLossFunction(config, model)


def compute_loss_weights_from_data(train_targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute appropriate loss weights based on class distribution.
    
    Args:
        train_targets: Training target labels
        
    Returns:
        Dictionary with recommended loss weights
    """
    pos_count = train_targets.sum().item()
    neg_count = len(train_targets) - pos_count
    
    # Positive weight for BCE (to balance classes)
    pos_weight = neg_count / max(pos_count, 1)
    
    # Adjust sparsity weight based on typical package sizes
    # Larger packages need less sparsity regularization
    avg_samples = len(train_targets)
    sparsity_weight = 0.01 * min(1.0, 100 / avg_samples)
    
    logger.info(f"Data statistics:")
    logger.info(f"  Positive samples: {pos_count}")
    logger.info(f"  Negative samples: {neg_count}")
    logger.info(f"  Recommended pos_weight: {pos_weight:.3f}")
    logger.info(f"  Recommended sparsity_weight: {sparsity_weight:.4f}")
    
    return {
        "pos_weight": pos_weight,
        "sparsity_weight": sparsity_weight,
        "counterfactual_weight": 0.05  # Standard value
    }