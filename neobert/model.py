"""
NeoBERT classifier model for malicious package detection.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .config import NeoBERTConfig
from .encoder import NeoBERTEncoder
from .pooling import create_pooling_layer, PoolingOutput
from .unit_processor import UnitProcessor, PackageUnit

logger = logging.getLogger(__name__)


@dataclass
class NeoBERTOutput:
    """Output from NeoBERT model."""
    
    # Main classification results
    logits: torch.Tensor
    probabilities: torch.Tensor
    prediction: int
    confidence: float
    
    # Auxiliary predictions
    api_predictions: Optional[torch.Tensor] = None
    phase_predictions: Optional[torch.Tensor] = None
    
    # Interpretability
    attention_weights: Optional[torch.Tensor] = None
    unit_scores: Optional[torch.Tensor] = None
    suspicious_units: Optional[List[int]] = None
    
    # Metadata
    num_units_processed: int = 0
    inference_time_seconds: float = 0.0
    pooling_metadata: Optional[Dict[str, Any]] = None


class NeoBERTClassifier(nn.Module):
    """Complete NeoBERT classifier for malicious package detection."""
    
    def __init__(self, config: NeoBERTConfig, device: str = "auto"):
        super().__init__()
        self.config = config
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize components
        self.unit_processor = UnitProcessor(config)
        self.encoder = NeoBERTEncoder(config)
        self.pooling = create_pooling_layer(config)
        
        # Main classifier
        self.classifier = self._build_classifier()
        
        # Auxiliary classifiers
        if config.use_api_prediction:
            self.api_classifier = self._build_api_classifier()
        else:
            self.api_classifier = None
            
        if config.use_phase_prediction:
            self.phase_classifier = self._build_phase_classifier()
        else:
            self.phase_classifier = None
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"NeoBERT classifier initialized on {self.device}")
        logger.info(f"Model parameters: {self.count_parameters():,}")
    
    def _build_classifier(self) -> nn.Module:
        """Build main binary classifier."""
        
        return nn.Sequential(
            nn.Linear(self.config.projection_dim, self.config.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(self.config.classifier_hidden_dim, self.config.classifier_hidden_dim // 2),
            nn.ReLU(), 
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(self.config.classifier_hidden_dim // 2, 1)
        )
    
    def _build_api_classifier(self) -> nn.Module:
        """Build auxiliary API prediction classifier."""
        
        return nn.Sequential(
            nn.Linear(self.config.projection_dim, self.config.classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.config.classifier_hidden_dim // 2, len(self.config.risky_apis))
        )
    
    def _build_phase_classifier(self) -> nn.Module:
        """Build auxiliary phase prediction classifier."""
        
        return nn.Sequential(
            nn.Linear(self.config.projection_dim, self.config.classifier_hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.config.classifier_hidden_dim // 4, 3)  # install, runtime, test
        )
    
    def forward(self, units: List[PackageUnit]) -> NeoBERTOutput:
        """Forward pass through the complete model."""
        
        start_time = time.time()
        
        if not units:
            return self._empty_output()
        
        # Encode units
        unit_embeddings = self.encoder(units)  # (num_units, embedding_dim)
        
        if unit_embeddings.size(0) == 0:
            return self._empty_output()
        
        # Pool to package level
        pooling_output = self.pooling(unit_embeddings)
        package_embedding = pooling_output.package_embedding
        
        # Main classification
        logits = self.classifier(package_embedding).squeeze(-1)  # Remove last dim if single value
        probabilities = torch.sigmoid(logits)
        
        # Handle both single package and batch cases
        if logits.dim() == 0:
            # Single package
            prediction = int(probabilities > 0.5)
            confidence = float(probabilities.detach() if prediction == 1 else 1 - probabilities.detach())
        else:
            # Batch of packages
            predictions = (probabilities > 0.5).int()
            prediction = predictions[0].item() if len(predictions) == 1 else predictions
            confidence = float(probabilities[0].detach() if prediction == 1 else 1 - probabilities[0].detach())
        
        # Auxiliary predictions
        api_predictions = None
        if self.api_classifier is not None:
            api_logits = self.api_classifier(package_embedding)
            api_predictions = torch.sigmoid(api_logits)
        
        phase_predictions = None
        if self.phase_classifier is not None:
            phase_logits = self.phase_classifier(package_embedding)
            phase_predictions = F.softmax(phase_logits, dim=-1)
        
        inference_time = time.time() - start_time
        
        return NeoBERTOutput(
            logits=logits,
            probabilities=probabilities,
            prediction=prediction,
            confidence=confidence,
            api_predictions=api_predictions,
            phase_predictions=phase_predictions,
            attention_weights=pooling_output.attention_weights,
            unit_scores=pooling_output.unit_scores,
            suspicious_units=pooling_output.suspicious_unit_indices,
            num_units_processed=len(units),
            inference_time_seconds=inference_time,
            pooling_metadata=pooling_output.pooling_metadata
        )
    
    def predict_package(self, 
                       package_name: str,
                       file_contents: Dict[str, str],
                       ecosystem: str = "unknown") -> NeoBERTOutput:
        """Predict maliciousness of a complete package."""
        
        self.eval()
        with torch.no_grad():
            # Process package into units
            units = self.unit_processor.process_package(
                package_name, file_contents, ecosystem
            )
            
            # Make prediction
            output = self.forward(units)
            
            return output
    
    def predict_from_units(self, units: List[PackageUnit]) -> NeoBERTOutput:
        """Predict from pre-processed units."""
        
        self.eval()
        with torch.no_grad():
            return self.forward(units)
    
    def get_detailed_explanation(self,
                               package_name: str,
                               file_contents: Dict[str, str],
                               ecosystem: str = "unknown") -> Dict[str, Any]:
        """Get detailed explanation of prediction."""
        
        # Process package
        units = self.unit_processor.process_package(package_name, file_contents, ecosystem)
        
        # Get prediction
        output = self.predict_from_units(units)
        
        # Build explanation
        explanation = {
            "package_name": package_name,
            "ecosystem": ecosystem,
            "prediction": {
                "is_malicious": output.prediction == 1,
                "confidence": output.confidence,
                "probability": float(output.probabilities),
                "decision_threshold": 0.5
            },
            "processing_stats": {
                "num_units": output.num_units_processed,
                "inference_time_seconds": output.inference_time_seconds,
                "pooling_strategy": self.config.pooling_strategy
            },
            "unit_analysis": self._analyze_units(units, output),
            "api_analysis": self._analyze_api_predictions(output, units),
            "phase_analysis": self._analyze_phase_predictions(output, units),
            "attention_analysis": self._analyze_attention(output, units),
        }
        
        # Add pooling-specific metadata
        if output.pooling_metadata:
            explanation["pooling_metadata"] = output.pooling_metadata
        
        return explanation
    
    def _analyze_units(self, units: List[PackageUnit], output: NeoBERTOutput) -> Dict[str, Any]:
        """Analyze individual units."""
        
        unit_analysis = {
            "total_units": len(units),
            "unit_types": {},
            "phase_distribution": {},
            "size_stats": {},
            "suspicious_units": []
        }
        
        if not units:
            return unit_analysis
        
        # Count types and phases
        for unit in units:
            unit_type = unit.unit_type
            phase = unit.phase_tag
            
            unit_analysis["unit_types"][unit_type] = unit_analysis["unit_types"].get(unit_type, 0) + 1
            unit_analysis["phase_distribution"][phase] = unit_analysis["phase_distribution"].get(phase, 0) + 1
        
        # Size statistics
        sizes = [unit.file_size for unit in units]
        unit_analysis["size_stats"] = {
            "min": min(sizes),
            "max": max(sizes),
            "mean": np.mean(sizes),
            "total": sum(sizes)
        }
        
        # Suspicious units
        if output.suspicious_units:
            for idx in output.suspicious_units:
                if idx < len(units):
                    unit = units[idx]
                    
                    suspicion_info = {
                        "unit_name": unit.unit_name,
                        "unit_type": unit.unit_type,
                        "source_file": unit.source_file,
                        "phase": unit.phase_tag,
                        "size": unit.file_size,
                        "entropy": unit.shannon_entropy,
                        "risky_apis": sum(unit.risky_api_counts.values()),
                        "api_details": unit.risky_api_counts
                    }
                    
                    # Add attention weight if available
                    if output.attention_weights is not None and idx < len(output.attention_weights):
                        suspicion_info["attention_weight"] = float(output.attention_weights[idx])
                    
                    # Add unit score if available (from MIL)
                    if output.unit_scores is not None and idx < len(output.unit_scores):
                        suspicion_info["suspicion_score"] = float(output.unit_scores[idx])
                    
                    unit_analysis["suspicious_units"].append(suspicion_info)
        
        return unit_analysis
    
    def _analyze_api_predictions(self, output: NeoBERTOutput, units: List[PackageUnit]) -> Dict[str, Any]:
        """Analyze API predictions."""
        
        analysis = {
            "predicted_apis": [],
            "actual_apis_found": {},
            "api_accuracy": None
        }
        
        if output.api_predictions is not None:
            # Get predicted APIs above threshold
            api_probs = output.api_predictions
            if api_probs.dim() > 1:
                api_probs = api_probs[0]  # Take first in batch
            
            for i, prob in enumerate(api_probs):
                if prob > self.config.api_prediction_threshold:
                    api_name = self.config.risky_apis[i]
                    analysis["predicted_apis"].append({
                        "api": api_name,
                        "probability": float(prob)
                    })
        
        # Count actual APIs found in units
        for unit in units:
            for api, count in unit.risky_api_counts.items():
                if count > 0:
                    analysis["actual_apis_found"][api] = analysis["actual_apis_found"].get(api, 0) + count
        
        # Simple accuracy calculation
        if analysis["predicted_apis"] and analysis["actual_apis_found"]:
            predicted_set = set(item["api"] for item in analysis["predicted_apis"])
            actual_set = set(analysis["actual_apis_found"].keys())
            
            intersection = len(predicted_set & actual_set)
            union = len(predicted_set | actual_set)
            
            analysis["api_accuracy"] = intersection / union if union > 0 else 0.0
        
        return analysis
    
    def _analyze_phase_predictions(self, output: NeoBERTOutput, units: List[PackageUnit]) -> Dict[str, Any]:
        """Analyze phase predictions."""
        
        analysis = {
            "predicted_distribution": None,
            "actual_distribution": {},
            "dominant_phase": None
        }
        
        if output.phase_predictions is not None:
            phase_probs = output.phase_predictions
            if phase_probs.dim() > 1:
                phase_probs = phase_probs[0]
            
            phase_names = ["install", "runtime", "test"]
            analysis["predicted_distribution"] = {
                phase_names[i]: float(phase_probs[i]) 
                for i in range(min(len(phase_names), len(phase_probs)))
            }
            
            # Dominant predicted phase
            max_idx = torch.argmax(phase_probs).item()
            if max_idx < len(phase_names):
                analysis["dominant_phase"] = phase_names[max_idx]
        
        # Actual phase distribution
        for unit in units:
            phase = unit.phase_tag
            analysis["actual_distribution"][phase] = analysis["actual_distribution"].get(phase, 0) + 1
        
        return analysis
    
    def _analyze_attention(self, output: NeoBERTOutput, units: List[PackageUnit]) -> Dict[str, Any]:
        """Analyze attention patterns."""
        
        analysis = {
            "attention_available": output.attention_weights is not None,
            "attention_entropy": None,
            "attention_concentration": None,
            "top_attended_units": []
        }
        
        if output.attention_weights is not None:
            weights = output.attention_weights
            if weights.dim() > 1:
                weights = weights[0]  # Take first in batch
            
            # Entropy calculation
            weights_norm = weights + 1e-8  # Avoid log(0)
            entropy = -(weights_norm * torch.log(weights_norm)).sum()
            max_entropy = np.log(len(weights)) if len(weights) > 0 else 1
            analysis["attention_entropy"] = float(entropy / max_entropy) if max_entropy > 0 else 0.0
            
            # Concentration (how much attention is on top units)
            if len(weights) > 0:
                top_3_weights = torch.topk(weights, min(3, len(weights))).values
                analysis["attention_concentration"] = float(top_3_weights.sum())
                
                # Top attended unit details
                top_indices = torch.topk(weights, min(5, len(weights))).indices
                for i, idx in enumerate(top_indices):
                    if idx < len(units):
                        unit = units[idx]
                        analysis["top_attended_units"].append({
                            "rank": i + 1,
                            "unit_name": unit.unit_name,
                            "attention_weight": float(weights[idx]),
                            "unit_type": unit.unit_type,
                            "phase": unit.phase_tag
                        })
        
        return analysis
    
    def _empty_output(self) -> NeoBERTOutput:
        """Create empty output for edge cases."""
        
        return NeoBERTOutput(
            logits=torch.tensor(0.0, device=self.device),
            probabilities=torch.tensor(0.5, device=self.device),
            prediction=0,
            confidence=0.5,
            num_units_processed=0,
            inference_time_seconds=0.0
        )
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        
        return {
            "model_type": "NeoBERT",
            "model_class": "Transformer-based Package Classifier",
            "approach": "Unit-level encoding with package-level pooling",
            "supports_explanations": True,
            "supports_unit_localization": self.config.pooling_strategy in ["attention", "mil"],
            "inference_method": "local",
            "device": str(self.device),
            
            "config": {
                "backbone_model": self.config.model_name,
                "pooling_strategy": self.config.pooling_strategy,
                "max_units_per_package": self.config.max_units_per_package,
                "max_tokens_per_unit": self.config.max_tokens_per_unit,
                "use_augmented_features": self.config.use_augmented_features,
                "projection_dim": self.config.projection_dim,
                "total_parameters": self.count_parameters()
            }
        }
    
    def save_pretrained(self, save_path: str):
        """Save the complete model."""
        
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_info': self.get_model_info()
        }, os.path.join(save_path, 'model.pth'))
        
        # Save encoder separately for potential reuse
        self.encoder.save_pretrained(os.path.join(save_path, 'encoder'))
        
        logger.info(f"NeoBERT model saved to {save_path}")
    
    @classmethod
    def load_pretrained(cls, load_path: str, device: str = "auto"):
        """Load a pre-trained model."""
        
        import os
        
        # Load model checkpoint
        model_path = os.path.join(load_path, 'model.pth')
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model
        config = checkpoint['config']
        model = cls(config, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"NeoBERT model loaded from {load_path}")
        return model


def create_neobert_model(config: NeoBERTConfig, device: str = "auto") -> NeoBERTClassifier:
    """Create NeoBERT model with given configuration."""
    
    model = NeoBERTClassifier(config, device)
    
    logger.info(f"Created NeoBERT model with {model.count_parameters():,} parameters")
    logger.info(f"Pooling strategy: {config.pooling_strategy}")
    logger.info(f"Device: {model.device}")
    
    return model