"""
CPG-GNN model implementation with Graph Neural Networks.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.attention import GlobalAttention

from .config import CPGConfig
from .feature_extractor import CPGFeatureExtractor
from .graph_builder import CodePropertyGraph

logger = logging.getLogger(__name__)


@dataclass
class CPGModelOutput:
    """Output from CPG-GNN model."""
    logits: torch.Tensor
    probabilities: torch.Tensor
    prediction: int
    confidence: float
    
    # Attention and interpretability
    attention_weights: Optional[torch.Tensor] = None
    top_suspicious_subgraphs: Optional[List[Dict]] = None
    
    # Auxiliary predictions
    api_predictions: Optional[torch.Tensor] = None
    entropy_prediction: Optional[torch.Tensor] = None
    
    # Metadata
    num_nodes: int = 0
    num_edges: int = 0


class CPGModel(nn.Module):
    """Code Property Graph with Graph Neural Networks for malicious package detection."""
    
    def __init__(self, config: CPGConfig, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.device = device
        
        # Feature extractor
        self.feature_extractor = CPGFeatureExtractor(config)
        
        # GNN layers
        self.gnn_layers = self._build_gnn_layers()
        
        # Pooling layer
        self.pooling = self._build_pooling_layer()
        
        # Classifier
        self.classifier = self._build_classifier()
        
        # Auxiliary heads
        self.api_classifier = nn.Linear(config.gnn_hidden_dim, len(config.risky_apis))
        self.entropy_regressor = nn.Linear(config.gnn_hidden_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(config.classifier_dropout)
        
        # Move to device
        self.to(device)
    
    def _build_gnn_layers(self) -> nn.ModuleList:
        """Build GNN layers based on configuration."""
        
        layers = nn.ModuleList()
        
        for i in range(self.config.num_gnn_layers):
            input_dim = self.config.node_embedding_dim if i == 0 else self.config.gnn_hidden_dim
            
            if self.config.gnn_type.lower() == "gin":
                # Graph Isomorphism Network
                mlp = nn.Sequential(
                    nn.Linear(input_dim, self.config.gnn_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.config.gnn_hidden_dim, self.config.gnn_hidden_dim)
                )
                conv = GINConv(mlp)
                
            elif self.config.gnn_type.lower() == "gat":
                # Graph Attention Network
                conv = GATConv(
                    input_dim,
                    self.config.gnn_hidden_dim // self.config.attention_heads,
                    heads=self.config.attention_heads,
                    dropout=self.config.gnn_dropout,
                    concat=True
                )
                
            else:
                # Default to GIN
                mlp = nn.Sequential(
                    nn.Linear(input_dim, self.config.gnn_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.config.gnn_hidden_dim, self.config.gnn_hidden_dim)
                )
                conv = GINConv(mlp)
            
            layers.append(conv)
        
        return layers
    
    def _build_pooling_layer(self):
        """Build pooling layer for graph-level representation."""
        
        if self.config.pooling_type == "attention":
            # Attention-based pooling
            attention_net = nn.Sequential(
                nn.Linear(self.config.gnn_hidden_dim, self.config.gnn_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.config.gnn_hidden_dim // 2, 1)
            )
            return GlobalAttention(attention_net)
            
        elif self.config.pooling_type == "mean":
            return global_mean_pool
            
        elif self.config.pooling_type == "max":
            return global_max_pool
            
        else:
            return global_add_pool
    
    def _build_classifier(self) -> nn.Module:
        """Build classification head."""
        
        # Input dimension includes global features
        input_dim = self.config.gnn_hidden_dim
        if self.config.use_metadata_features:
            input_dim += len(self._get_expected_global_features())
        
        return nn.Sequential(
            nn.Linear(input_dim, self.config.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(self.config.classifier_hidden_dim, self.config.classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(self.config.classifier_hidden_dim // 2, self.config.num_classes)
        )
    
    def forward(self, data: Data) -> CPGModelOutput:
        """Forward pass through the CPG-GNN model."""
        
        # Handle batch or single graph
        if isinstance(data, Batch):
            batch = data
        else:
            batch = Batch.from_data_list([data])
        
        # Get node features
        x = batch.x
        edge_index = batch.edge_index
        batch_idx = batch.batch
        
        # Apply GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.config.gnn_dropout, training=self.training)
        
        # Graph-level pooling
        if self.config.pooling_type == "attention":
            graph_embedding, attention_weights = self.pooling(x, batch_idx, return_attention_weights=True)
        else:
            graph_embedding = self.pooling(x, batch_idx)
            attention_weights = None
        
        # Add global features if available
        if hasattr(batch, 'global_features') and batch.global_features is not None:
            # Handle batch dimension for global features
            if len(batch.global_features.shape) == 1:
                global_features = batch.global_features.unsqueeze(0)
            else:
                global_features = batch.global_features
                
            graph_embedding = torch.cat([graph_embedding, global_features], dim=-1)
        
        # Classification
        logits = self.classifier(graph_embedding)
        probabilities = F.softmax(logits, dim=-1)
        
        # Get predictions
        prediction = torch.argmax(probabilities, dim=-1)
        confidence = torch.max(probabilities, dim=-1).values
        
        # Auxiliary predictions
        api_predictions = self.api_classifier(x.mean(dim=0)) if x.size(0) > 0 else None
        entropy_prediction = self.entropy_regressor(x.mean(dim=0)) if x.size(0) > 0 else None
        
        # Suspicious subgraph detection
        top_subgraphs = None
        if attention_weights is not None:
            top_subgraphs = self._identify_suspicious_subgraphs(
                batch, x, attention_weights
            )
        
        return CPGModelOutput(
            logits=logits,
            probabilities=probabilities,
            prediction=prediction.item() if prediction.dim() == 0 else prediction[0].item(),
            confidence=confidence.item() if confidence.dim() == 0 else confidence[0].item(),
            attention_weights=attention_weights,
            top_suspicious_subgraphs=top_subgraphs,
            api_predictions=api_predictions,
            entropy_prediction=entropy_prediction,
            num_nodes=x.size(0),
            num_edges=edge_index.size(1)
        )
    
    def predict_package(self, cpg: CodePropertyGraph) -> CPGModelOutput:
        """Make prediction on a single package."""
        
        self.eval()
        with torch.no_grad():
            # Extract features
            data = self.feature_extractor.extract_features(cpg)
            data = data.to(self.device)
            
            # Forward pass
            output = self.forward(data)
            
            return output
    
    def _identify_suspicious_subgraphs(self, batch: Batch, node_features: torch.Tensor, 
                                     attention_weights: torch.Tensor) -> List[Dict]:
        """Identify top-k suspicious subgraphs based on attention weights."""
        
        if attention_weights is None or node_features.size(0) == 0:
            return []
        
        # Get top-k nodes by attention
        k = min(self.config.attention_heads, node_features.size(0))
        top_k_indices = torch.topk(attention_weights.squeeze(), k).indices
        
        subgraphs = []
        for i, node_idx in enumerate(top_k_indices):
            node_idx = node_idx.item()
            
            subgraph_info = {
                "rank": i + 1,
                "node_index": node_idx,
                "attention_weight": attention_weights[node_idx].item(),
                "node_type": "unknown",
                "suspicious_apis": [],
                "code_snippet": ""
            }
            
            # Get node information from original data
            if hasattr(batch, 'node_mapping'):
                original_node_id = None
                for orig_id, mapped_id in batch.node_mapping.items():
                    if mapped_id == node_idx:
                        original_node_id = orig_id
                        break
                
                if original_node_id is not None:
                    # This would require access to the original CPG
                    # For now, just include basic information
                    subgraph_info["original_node_id"] = original_node_id
            
            subgraphs.append(subgraph_info)
        
        return subgraphs
    
    def get_attention_explanation(self, cpg: CodePropertyGraph) -> Dict[str, Any]:
        """Get detailed attention-based explanation for a package."""
        
        output = self.predict_package(cpg)
        
        explanation = {
            "package_name": cpg.package_name,
            "ecosystem": cpg.ecosystem,
            "prediction": {
                "is_malicious": output.prediction == 1,
                "confidence": output.confidence,
                "probability": output.probabilities[1].item() if output.probabilities.size(0) > 1 else 0.0
            },
            "graph_stats": {
                "num_nodes": output.num_nodes,
                "num_edges": output.num_edges,
                "num_files": cpg.num_files
            },
            "suspicious_subgraphs": output.top_suspicious_subgraphs or [],
            "api_analysis": self._analyze_api_predictions(output.api_predictions, cpg),
            "attention_analysis": self._analyze_attention_patterns(output.attention_weights)
        }
        
        return explanation
    
    def _analyze_api_predictions(self, api_predictions: Optional[torch.Tensor], 
                                cpg: CodePropertyGraph) -> Dict[str, Any]:
        """Analyze auxiliary API predictions."""
        
        if api_predictions is None:
            return {"error": "No API predictions available"}
        
        # Get top predicted APIs
        api_probs = F.sigmoid(api_predictions)
        top_apis = torch.topk(api_probs, min(5, len(self.config.risky_apis)))
        
        predicted_apis = []
        for i, prob in zip(top_apis.indices, top_apis.values):
            api_name = self.config.risky_apis[i.item()]
            predicted_apis.append({
                "api": api_name,
                "probability": prob.item(),
                "found_in_code": api_name in cpg.api_calls
            })
        
        return {
            "predicted_risky_apis": predicted_apis,
            "actual_apis_found": list(cpg.api_calls),
            "api_diversity": len(cpg.api_calls) / len(self.config.risky_apis)
        }
    
    def _analyze_attention_patterns(self, attention_weights: Optional[torch.Tensor]) -> Dict[str, Any]:
        """Analyze attention weight patterns."""
        
        if attention_weights is None:
            return {"error": "No attention weights available"}
        
        weights = attention_weights.squeeze()
        
        return {
            "attention_entropy": self._calculate_attention_entropy(weights),
            "max_attention": weights.max().item(),
            "attention_concentration": (weights > weights.mean() + weights.std()).sum().item(),
            "attention_distribution": {
                "mean": weights.mean().item(),
                "std": weights.std().item(),
                "min": weights.min().item(),
                "max": weights.max().item()
            }
        }
    
    def _calculate_attention_entropy(self, weights: torch.Tensor) -> float:
        """Calculate entropy of attention distribution."""
        if weights.numel() == 0:
            return 0.0
        
        # Normalize to probabilities
        probs = F.softmax(weights, dim=-1)
        
        # Calculate entropy
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        
        # Normalize by max possible entropy
        max_entropy = math.log(weights.numel())
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy.item()
    
    def _get_expected_global_features(self) -> List[str]:
        """Get list of expected global feature names."""
        return [
            "node_count", "edge_count", "file_count",
            "ast_edges", "cfg_edges", "dfg_edges", "inter_file_edges", "metadata_edges", "unknown_edges",
            "api_diversity", "ecosystem_npm", "ecosystem_pypi"
        ]


def create_cpg_model(config: CPGConfig, device: str = "auto") -> CPGModel:
    """Create CPG-GNN model with default configuration."""
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = CPGModel(config, device)
    
    logger.info(f"Created CPG-GNN model with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Model device: {device}")
    
    return model