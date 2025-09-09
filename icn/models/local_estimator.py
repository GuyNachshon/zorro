"""
Local Intent Estimator for ICN.
Transforms code units into intent distributions and embeddings using CodeBERT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class LocalIntentOutput:
    """Output from the Local Intent Estimator."""
    fixed_intent_dist: torch.Tensor  # [batch_size, n_fixed_intents]
    latent_intent_dist: torch.Tensor  # [batch_size, n_latent_intents]
    unit_embeddings: torch.Tensor  # [batch_size, embedding_dim]
    attention_weights: Optional[torch.Tensor] = None


class LocalIntentEstimator(nn.Module):
    """
    Local Intent Estimator using transformer architecture.
    Processes individual code units to extract intent distributions.
    """
    
    def __init__(
        self,
        vocab_size: int = 50265,  # CodeBERT vocabulary size
        n_fixed_intents: int = 15,
        n_latent_intents: int = 10,
        embedding_dim: int = 768,
        hidden_dim: int = 768,
        n_layers: int = 6,
        n_heads: int = 12,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        use_pretrained: bool = True,
        model_name: str = "microsoft/codebert-base"
    ):
        super().__init__()
        
        self.n_fixed_intents = n_fixed_intents
        self.n_latent_intents = n_latent_intents
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        if use_pretrained:
            # Use pretrained CodeBERT
            from transformers import AutoModel
            self.encoder = AutoModel.from_pretrained(model_name)
            self.embedding_dim = self.encoder.config.hidden_size
        else:
            # Build transformer from scratch
            self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.position_embeddings = nn.Embedding(max_seq_length, embedding_dim)
            self.token_type_embeddings = nn.Embedding(2, embedding_dim)  # code vs comment
            
            # Transformer encoder layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Additional feature embeddings
        self.phase_embedding = nn.Embedding(3, 64)  # install, postinstall, runtime
        self.api_category_embedding = nn.Linear(15, 128)  # 15 API categories
        self.ast_feature_embedding = nn.Linear(50, 64)  # AST node type features
        
        # Feature fusion layer
        feature_dim = self.embedding_dim + 64 + 128 + 64  # encoder + phase + api + ast
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Intent classification heads
        self.fixed_intent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_fixed_intents)
        )
        
        self.latent_intent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_latent_intents)
        )
        
        # Unit embedding projection
        self.embedding_projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Latent intent discovery (contrastive learning)
        self.latent_prototypes = nn.Parameter(
            torch.randn(n_latent_intents, embedding_dim)
        )
        
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
    def forward(
        self,
        input_ids: torch.Tensor,  # [batch_size, seq_length]
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        phase_ids: Optional[torch.Tensor] = None,  # [batch_size]
        api_features: Optional[torch.Tensor] = None,  # [batch_size, 15]
        ast_features: Optional[torch.Tensor] = None,  # [batch_size, 50]
        return_attention: bool = False
    ) -> LocalIntentOutput:
        """
        Forward pass of the Local Intent Estimator.
        
        Args:
            input_ids: Tokenized code input
            attention_mask: Mask for valid tokens
            token_type_ids: Token type IDs (code vs comment)
            phase_ids: Phase indicator (install/runtime)
            api_features: Binary vector of API category presence
            ast_features: AST node type features
            return_attention: Whether to return attention weights
            
        Returns:
            LocalIntentOutput with intent distributions and embeddings
        """
        batch_size = input_ids.size(0)
        
        # Encode tokens
        if hasattr(self, 'token_embeddings'):
            # Custom transformer
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            token_embeds = self.token_embeddings(input_ids)
            position_embeds = self.position_embeddings(position_ids)
            
            if token_type_ids is not None:
                type_embeds = self.token_type_embeddings(token_type_ids)
                embeddings = token_embeds + position_embeds + type_embeds
            else:
                embeddings = token_embeds + position_embeds
            
            # Apply transformer encoder
            # PyTorch transformer expects key_padding_mask to be True for padded positions
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = ~attention_mask.bool()
            encoded = self.encoder(embeddings, src_key_padding_mask=key_padding_mask)
            
            # Pool to get sequence representation
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                encoded_masked = encoded * mask_expanded
                pooled = encoded_masked.sum(1) / mask_expanded.sum(1)
            else:
                pooled = encoded.mean(1)
        else:
            # Use pretrained CodeBERT
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True
            )
            pooled = outputs.pooler_output  # CLS token representation
        
        # Process additional features
        features = [pooled]
        
        if phase_ids is not None:
            phase_embeds = self.phase_embedding(phase_ids)
            features.append(phase_embeds)
        else:
            features.append(torch.zeros(batch_size, 64, device=pooled.device))
        
        if api_features is not None:
            api_embeds = self.api_category_embedding(api_features.float())
            features.append(api_embeds)
        else:
            features.append(torch.zeros(batch_size, 128, device=pooled.device))
        
        if ast_features is not None:
            ast_embeds = self.ast_feature_embedding(ast_features.float())
            features.append(ast_embeds)
        else:
            features.append(torch.zeros(batch_size, 64, device=pooled.device))
        
        # Fuse all features
        fused_features = torch.cat(features, dim=-1)
        hidden = self.feature_fusion(fused_features)
        
        # Generate intent distributions
        fixed_intent_logits = self.fixed_intent_head(hidden)
        fixed_intent_dist = F.softmax(fixed_intent_logits, dim=-1)
        
        latent_intent_logits = self.latent_intent_head(hidden)
        latent_intent_dist = F.softmax(latent_intent_logits, dim=-1)
        
        # Generate unit embeddings
        unit_embeddings = self.embedding_projection(hidden)
        
        # Optionally compute attention to latent prototypes
        if self.n_latent_intents > 0:
            # Contrastive learning with latent prototypes
            unit_norm = F.normalize(unit_embeddings, dim=-1)
            proto_norm = F.normalize(self.latent_prototypes, dim=-1)
            
            # Compute similarity to prototypes
            similarity = torch.matmul(unit_norm, proto_norm.T) / self.temperature
            latent_intent_dist = F.softmax(similarity, dim=-1)
        
        return LocalIntentOutput(
            fixed_intent_dist=fixed_intent_dist,
            latent_intent_dist=latent_intent_dist,
            unit_embeddings=unit_embeddings,
            attention_weights=None  # Could add attention visualization later
        )
    
    def compute_intent_entropy(self, intent_dist: torch.Tensor) -> torch.Tensor:
        """Compute entropy of intent distribution for uncertainty estimation."""
        eps = 1e-8
        entropy = -(intent_dist * torch.log(intent_dist + eps)).sum(dim=-1)
        return entropy
    
    def get_dominant_intents(
        self, 
        fixed_dist: torch.Tensor, 
        latent_dist: torch.Tensor,
        threshold: float = 0.1
    ) -> Tuple[List[int], List[int]]:
        """Extract dominant intent categories above threshold."""
        fixed_dominant = (fixed_dist > threshold).nonzero(as_tuple=True)[1].tolist()
        latent_dominant = (latent_dist > threshold).nonzero(as_tuple=True)[1].tolist()
        return fixed_dominant, latent_dominant


class IntentVocabulary:
    """Maps between intent names and indices."""
    
    def __init__(self):
        self.fixed_intents = [
            "net.outbound",
            "net.inbound", 
            "fs.read",
            "fs.write",
            "proc.spawn",
            "eval",
            "crypto",
            "sys.env",
            "installer",
            "encoding",
            "config",
            "logging",
            "database",
            "auth",
            "benign"
        ]
        
        self.intent_to_idx = {intent: i for i, intent in enumerate(self.fixed_intents)}
        self.idx_to_intent = {i: intent for i, intent in enumerate(self.fixed_intents)}
    
    def encode(self, intents: List[str]) -> torch.Tensor:
        """Convert intent names to one-hot vector."""
        vector = torch.zeros(len(self.fixed_intents))
        for intent in intents:
            if intent in self.intent_to_idx:
                vector[self.intent_to_idx[intent]] = 1.0
        return vector
    
    def decode(self, vector: torch.Tensor, threshold: float = 0.1) -> List[str]:
        """Convert intent vector back to names."""
        intents = []
        for i, score in enumerate(vector):
            if score > threshold:
                intents.append(self.idx_to_intent[i])
        return intents


if __name__ == "__main__":
    # Test the Local Intent Estimator
    print("Testing Local Intent Estimator...")
    
    # Create model
    model = LocalIntentEstimator(
        use_pretrained=False,  # Use custom transformer for testing
        n_layers=2,  # Smaller for testing
        hidden_dim=256
    )
    
    # Create dummy inputs
    batch_size = 4
    seq_length = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.bool)
    phase_ids = torch.randint(0, 3, (batch_size,))
    api_features = torch.randn(batch_size, 15)
    ast_features = torch.randn(batch_size, 50)
    
    # Forward pass
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        phase_ids=phase_ids,
        api_features=api_features,
        ast_features=ast_features
    )
    
    print(f"Fixed intent distribution shape: {output.fixed_intent_dist.shape}")
    print(f"Latent intent distribution shape: {output.latent_intent_dist.shape}")
    print(f"Unit embeddings shape: {output.unit_embeddings.shape}")
    
    # Test intent vocabulary
    vocab = IntentVocabulary()
    test_intents = ["net.outbound", "crypto", "eval"]
    encoded = vocab.encode(test_intents)
    decoded = vocab.decode(encoded)
    print(f"\nIntent encoding test:")
    print(f"  Original: {test_intents}")
    print(f"  Encoded: {encoded.nonzero(as_tuple=True)[0].tolist()}")
    print(f"  Decoded: {decoded}")
    
    print("\n✅ Local Intent Estimator implemented successfully!")