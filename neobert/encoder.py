"""
NeoBERT encoder for unit-level embeddings.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig

from .config import NeoBERTConfig
from .unit_processor import PackageUnit

logger = logging.getLogger(__name__)


class NeoBERTEncoder(nn.Module):
    """NeoBERT encoder for generating unit-level embeddings."""
    
    def __init__(self, config: NeoBERTConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained model
        self._load_pretrained_model()
        
        # Feature processing layers
        self._build_feature_layers()
        
        # Projection layer
        self.projection = nn.Linear(
            self.embedding_dim + self.augmented_feature_dim,
            config.projection_dim
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.projection_dim)
    
    def _load_pretrained_model(self):
        """Load pre-trained NeoBERT/CodeBERT model."""
        
        try:
            # Try to load the specified model
            self.model_config = AutoConfig.from_pretrained(self.config.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.transformer = AutoModel.from_pretrained(self.config.model_name)
            
            logger.info(f"Loaded model: {self.config.model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load {self.config.model_name}: {e}")
            
            # Fallback to CodeBERT
            fallback_model = "microsoft/codebert-base"
            logger.info(f"Falling back to: {fallback_model}")
            
            self.model_config = AutoConfig.from_pretrained(fallback_model)
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.transformer = AutoModel.from_pretrained(fallback_model)
            
            # Update config
            self.config.model_name = fallback_model
        
        # Get embedding dimension
        self.embedding_dim = self.model_config.hidden_size
        
        # Freeze encoder if specified
        if self.config.freeze_encoder:
            for param in self.transformer.parameters():
                param.requires_grad = False
            logger.info("Frozen NeoBERT encoder weights")
    
    def _build_feature_layers(self):
        """Build layers for processing augmented features."""
        
        if not self.config.use_augmented_features:
            self.augmented_feature_dim = 0
            return
        
        # API feature embedding
        self.api_embedding = nn.Embedding(
            len(self.config.risky_apis) + 1,  # +1 for "none"
            self.config.api_feature_dim
        )
        
        # Phase embedding  
        phase_vocab_size = 4  # install, runtime, test, unknown
        self.phase_embedding = nn.Embedding(
            phase_vocab_size,
            self.config.phase_feature_dim
        )
        
        # Continuous feature processing
        self.continuous_features = nn.Linear(
            4,  # entropy, file_size, import_count, token_count
            self.config.metadata_feature_dim
        )
        
        # Calculate total augmented feature dimension
        self.augmented_feature_dim = (
            len(self.config.risky_apis) +  # API counts (one-hot style)
            self.config.api_feature_dim +  # API embedding
            self.config.phase_feature_dim +  # Phase embedding
            self.config.metadata_feature_dim +  # Continuous features
            self.config.entropy_feature_bins  # Entropy histogram
        )
        
        # Phase vocabulary mapping
        self.phase_to_id = {
            "install": 0,
            "runtime": 1, 
            "test": 2,
            "unknown": 3
        }
    
    def forward(self, units: List[PackageUnit]) -> torch.Tensor:
        """Forward pass to generate unit embeddings."""
        
        if not units:
            return torch.empty(0, self.config.projection_dim, device=self.get_device())
        
        # Get NeoBERT embeddings
        code_embeddings = self._get_code_embeddings(units)
        
        if self.config.use_augmented_features:
            # Get augmented features
            augmented_features = self._get_augmented_features(units)
            
            # Concatenate code and augmented features
            combined_embeddings = torch.cat([code_embeddings, augmented_features], dim=-1)
        else:
            combined_embeddings = code_embeddings
        
        # Project to target dimension
        projected = self.projection(combined_embeddings)
        projected = self.dropout(projected)
        projected = self.layer_norm(projected)
        
        return projected
    
    def _get_code_embeddings(self, units: List[PackageUnit]) -> torch.Tensor:
        """Get NeoBERT embeddings for code units."""
        
        embeddings = []
        
        # Process units in batches for efficiency
        # With 2 GPUs (DataParallel), can handle larger batches
        batch_size = 16  # Medium batches for efficiency
        for i in range(0, len(units), batch_size):
            batch_units = units[i:i + batch_size]
            batch_embeddings = self._process_batch(batch_units)
            embeddings.append(batch_embeddings)
        
        return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, self.embedding_dim)
    
    def _process_batch(self, units: List[PackageUnit]) -> torch.Tensor:
        """Process a batch of units through NeoBERT."""
        
        # Prepare input sequences
        input_texts = []
        for unit in units:
            # Use raw content, tokenizer will handle truncation
            text = unit.raw_content
            if not text.strip():
                text = "[EMPTY]"  # Handle empty units
            input_texts.append(text)
        
        # Tokenize batch
        try:
            inputs = self.tokenizer(
                input_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_tokens_per_unit,
                return_tensors="pt",
                return_attention_mask=True
            )
            
            # Move to device
            device = self.get_device()
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass through transformer
            with torch.set_grad_enabled(self.training):
                outputs = self.transformer(**inputs)
                
                # Use [CLS] token embedding (first token)
                embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Return zero embeddings as fallback
            return torch.zeros(len(units), self.embedding_dim, device=self.get_device())
    
    def _get_augmented_features(self, units: List[PackageUnit]) -> torch.Tensor:
        """Extract augmented features for units."""
        
        batch_size = len(units)
        device = self.get_device()
        
        features_list = []
        
        for unit in units:
            unit_features = []
            
            # 1. API counts (binary indicators)
            api_indicators = torch.zeros(len(self.config.risky_apis), device=device)
            for i, api in enumerate(self.config.risky_apis):
                if api in unit.risky_api_counts and unit.risky_api_counts[api] > 0:
                    api_indicators[i] = 1.0
            unit_features.append(api_indicators)
            
            # 2. API embedding (most frequent API)
            most_frequent_api = self._get_most_frequent_api(unit.risky_api_counts)
            api_id = self._get_api_id(most_frequent_api)
            api_emb = self.api_embedding(torch.tensor(api_id, device=device))
            unit_features.append(api_emb)
            
            # 3. Phase embedding
            phase_id = self.phase_to_id.get(unit.phase_tag, 3)  # 3 = unknown
            phase_emb = self.phase_embedding(torch.tensor(phase_id, device=device))
            unit_features.append(phase_emb)
            
            # 4. Continuous features
            continuous = torch.tensor([
                min(unit.shannon_entropy, 10.0) / 10.0,  # Normalized entropy
                min(unit.file_size, 50000) / 50000,  # Normalized file size
                min(unit.import_count, 100) / 100,  # Normalized import count
                min(len(unit.token_ids), 512) / 512  # Normalized token count
            ], device=device, dtype=torch.float)
            
            continuous_features = self.continuous_features(continuous)
            unit_features.append(continuous_features)
            
            # 5. Entropy histogram
            entropy_hist = self._compute_entropy_histogram(unit.raw_content)
            unit_features.append(torch.tensor(entropy_hist, device=device, dtype=torch.float))
            
            # Concatenate all features for this unit
            unit_feature_vector = torch.cat(unit_features, dim=0)
            features_list.append(unit_feature_vector)
        
        # Stack into batch tensor
        return torch.stack(features_list, dim=0)
    
    def _get_most_frequent_api(self, api_counts: Dict[str, int]) -> Optional[str]:
        """Get the most frequently used risky API."""
        if not api_counts:
            return None
        
        max_count = 0
        most_frequent = None
        
        for api, count in api_counts.items():
            if count > max_count:
                max_count = count
                most_frequent = api
        
        return most_frequent
    
    def _get_api_id(self, api_name: Optional[str]) -> int:
        """Get ID for API name."""
        if api_name is None or api_name not in self.config.risky_apis:
            return len(self.config.risky_apis)  # "none" ID
        
        return self.config.risky_apis.index(api_name)
    
    def _compute_entropy_histogram(self, text: str, bins: int = None) -> List[float]:
        """Compute entropy histogram of text."""
        
        if bins is None:
            bins = self.config.entropy_feature_bins
        
        if not text:
            return [0.0] * bins
        
        # Compute entropy for sliding windows
        window_size = max(50, len(text) // bins)
        entropies = []
        
        for i in range(0, len(text) - window_size + 1, window_size):
            window = text[i:i + window_size]
            entropy = self._compute_window_entropy(window)
            entropies.append(entropy)
        
        # Create histogram
        if not entropies:
            return [0.0] * bins
        
        hist, _ = np.histogram(entropies, bins=bins, range=(0, 8), density=True)
        return hist.tolist()
    
    def _compute_window_entropy(self, text: str) -> float:
        """Compute Shannon entropy of a text window."""
        
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        text_len = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            prob = count / text_len
            entropy -= prob * np.log2(prob)
        
        return entropy
    
    def get_device(self) -> torch.device:
        """Get the device the model is on."""
        return next(self.parameters()).device
    
    def encode_single_unit(self, unit: PackageUnit) -> torch.Tensor:
        """Encode a single unit (useful for inference)."""
        
        self.eval()
        with torch.no_grad():
            embedding = self.forward([unit])
            return embedding.squeeze(0)  # Remove batch dimension
    
    def encode_text(self, text: str, unit_type: str = "file") -> torch.Tensor:
        """Encode raw text as a unit (utility function)."""
        
        # Create minimal PackageUnit
        unit = PackageUnit(
            unit_id="temp",
            unit_name="temp_unit",
            unit_type=unit_type,
            source_file="temp.txt",
            raw_content=text,
            phase_tag="runtime"
        )
        
        # Extract basic features
        unit.risky_api_counts = self._extract_api_counts(text)
        
        return self.encode_single_unit(unit)
    
    def _extract_api_counts(self, text: str) -> Dict[str, int]:
        """Extract API counts from text (simplified version)."""
        
        api_counts = {}
        text_lower = text.lower()
        
        for api in self.config.risky_apis:
            count = text_lower.count(api.lower())
            if count > 0:
                api_counts[api] = count
        
        return api_counts
    
    def get_embedding_dimension(self) -> int:
        """Get the output embedding dimension."""
        return self.config.projection_dim
    
    def save_pretrained(self, save_path: str):
        """Save the encoder model."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save the full model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_config': self.model_config
        }, os.path.join(save_path, 'encoder.pth'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Encoder saved to {save_path}")
    
    @classmethod
    def load_pretrained(cls, load_path: str, device: str = "auto"):
        """Load a pre-trained encoder."""
        
        import os
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model checkpoint
        checkpoint_path = os.path.join(load_path, 'encoder.pth')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create encoder
        config = checkpoint['config']
        encoder = cls(config)
        encoder.load_state_dict(checkpoint['model_state_dict'])
        encoder.to(device)
        
        logger.info(f"Encoder loaded from {load_path}")
        return encoder