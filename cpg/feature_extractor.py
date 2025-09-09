"""
Feature extraction for CPG nodes and edges.
"""

import hashlib
import logging
import math
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import Data

from .config import CPGConfig
from .graph_builder import CodePropertyGraph

logger = logging.getLogger(__name__)


class CPGFeatureExtractor(nn.Module):
    """Extract features for CPG nodes and edges."""
    
    def __init__(self, config: CPGConfig):
        super().__init__()
        self.config = config
        
        # Initialize CodeBERT for semantic embeddings
        if config.use_pretrained_embeddings:
            self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
            self.code_model = AutoModel.from_pretrained(config.pretrained_model_name)
            
            # Freeze CodeBERT weights initially
            for param in self.code_model.parameters():
                param.requires_grad = False
        else:
            self.tokenizer = None
            self.code_model = None
        
        # Node type embedding
        self.node_type_embedding = nn.Embedding(
            config.node_type_vocab_size,
            config.node_embedding_dim // 4
        )
        
        # Edge type embedding
        self.edge_type_embedding = nn.Embedding(
            config.edge_type_vocab_size,
            config.node_embedding_dim // 8
        )
        
        # API embedding for risky APIs
        self.api_embedding = nn.Embedding(
            len(config.risky_apis) + 1,  # +1 for unknown
            config.node_embedding_dim // 8
        )
        
        # Metadata embedding
        if config.use_metadata_features:
            self.metadata_projector = nn.Linear(
                config.metadata_embedding_dim,
                config.node_embedding_dim
            )
        
        # Feature projection
        feature_dim = config.node_embedding_dim
        if config.use_pretrained_embeddings:
            feature_dim += 768  # CodeBERT dimension
        
        self.feature_projector = nn.Linear(feature_dim, config.node_embedding_dim)
        
        # Create vocabulary mappings
        self.node_type_to_id = {}
        self.edge_type_to_id = {
            'ast': 0, 'cfg': 1, 'dfg': 2, 'inter_file': 3, 'metadata_link': 4, 'unknown': 5
        }
        self.api_to_id = {api: i for i, api in enumerate(config.risky_apis)}
        self.api_to_id['unknown'] = len(self.api_to_id)
    
    def extract_features(self, cpg: CodePropertyGraph) -> Data:
        """Extract features for all nodes and edges in CPG."""
        
        # Convert CPG to PyTorch Geometric format
        data = cpg.to_pyg_data()
        
        # Extract node features
        node_features = self._extract_node_features(cpg)
        data.x = node_features
        
        # Extract edge features if enabled
        if self.config.use_edge_features:
            edge_features = self._extract_edge_features(cpg, data.edge_attr)
            data.edge_attr = edge_features
        
        # Add global features
        global_features = self._extract_global_features(cpg)
        data.global_features = global_features
        
        return data
    
    def _extract_node_features(self, cpg: CodePropertyGraph) -> torch.Tensor:
        """Extract features for all nodes in the graph."""
        
        node_features = []
        
        for node_id in cpg.graph.nodes():
            node_data = cpg.graph.nodes[node_id]
            
            # Get node text and type
            node_text = cpg.node_to_code.get(node_id, node_data.get('text', ''))
            node_type = cpg.node_to_type.get(node_id, node_data.get('type', 'unknown'))
            
            # Extract individual feature components
            features = []
            
            # 1. Semantic embedding from CodeBERT
            if self.config.use_pretrained_embeddings and self.code_model is not None:
                semantic_features = self._get_semantic_embedding(node_text)
                features.append(semantic_features)
            
            # 2. Node type embedding
            type_id = self._get_node_type_id(node_type)
            type_features = self.node_type_embedding(torch.tensor(type_id))
            features.append(type_features)
            
            # 3. API features
            api_features = self._get_api_features(node_text)
            features.append(api_features)
            
            # 4. Structural features
            structural_features = self._get_structural_features(cpg, node_id)
            features.append(structural_features)
            
            # 5. Text features
            text_features = self._get_text_features(node_text)
            features.append(text_features)
            
            # Concatenate all features
            node_feature_vector = torch.cat(features, dim=-1)
            node_features.append(node_feature_vector)
        
        # Stack into tensor
        if node_features:
            node_features_tensor = torch.stack(node_features)
            
            # Project to target dimension
            node_features_tensor = self.feature_projector(node_features_tensor)
            
            return node_features_tensor
        else:
            # Return empty tensor if no nodes
            return torch.empty((0, self.config.node_embedding_dim))
    
    def _get_semantic_embedding(self, text: str) -> torch.Tensor:
        """Get semantic embedding from CodeBERT."""
        
        if not text or not text.strip():
            return torch.zeros(768)
        
        try:
            # Tokenize and encode
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding='max_length'
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.code_model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[0, 0, :]  # First token
                
            return embedding
            
        except Exception as e:
            logger.warning(f"Error getting semantic embedding: {e}")
            return torch.zeros(768)
    
    def _get_api_features(self, text: str) -> torch.Tensor:
        """Extract API-related features."""
        
        # Check for risky APIs
        found_apis = []
        for api in self.config.risky_apis:
            if api.lower() in text.lower():
                found_apis.append(api)
        
        if found_apis:
            # Use first found API
            api_id = self.api_to_id.get(found_apis[0], self.api_to_id['unknown'])
        else:
            api_id = self.api_to_id['unknown']
        
        api_embedding = self.api_embedding(torch.tensor(api_id))
        
        # Add binary features for API categories
        api_flags = torch.zeros(len(self.config.risky_apis))
        for i, api in enumerate(self.config.risky_apis):
            if api.lower() in text.lower():
                api_flags[i] = 1.0
        
        return torch.cat([api_embedding, api_flags])
    
    def _get_structural_features(self, cpg: CodePropertyGraph, node_id: int) -> torch.Tensor:
        """Extract structural features for a node."""
        
        features = []
        
        # Node degree features
        in_degree = cpg.graph.in_degree(node_id)
        out_degree = cpg.graph.out_degree(node_id)
        total_degree = in_degree + out_degree
        
        # Normalize degrees
        max_degree = max(dict(cpg.graph.degree()).values()) if cpg.graph.nodes() else 1
        norm_in = in_degree / max_degree
        norm_out = out_degree / max_degree
        norm_total = total_degree / max_degree
        
        features.extend([norm_in, norm_out, norm_total])
        
        # Edge type distribution
        edge_type_counts = {}
        for _, _, edge_data in cpg.graph.edges(node_id, data=True):
            edge_type = edge_data.get('edge_type', 'unknown')
            edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        
        # Normalize edge type counts
        total_edges = sum(edge_type_counts.values()) if edge_type_counts else 1
        for edge_type in self.edge_type_to_id.keys():
            count = edge_type_counts.get(edge_type, 0)
            features.append(count / total_edges)
        
        # Distance to metadata node (if exists)
        metadata_nodes = [n for n, d in cpg.graph.nodes(data=True) 
                         if d.get('type') == 'metadata']
        if metadata_nodes:
            try:
                import networkx as nx
                distance = nx.shortest_path_length(cpg.graph, metadata_nodes[0], node_id)
                features.append(min(distance / 10.0, 1.0))  # Normalize
            except:
                features.append(1.0)  # Max distance if no path
        else:
            features.append(1.0)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _get_text_features(self, text: str) -> torch.Tensor:
        """Extract text-based features."""
        
        features = []
        
        if not text or not text.strip():
            return torch.zeros(5)
        
        # Text length (normalized)
        features.append(min(len(text) / 1000.0, 1.0))
        
        # Shannon entropy
        entropy = self._calculate_entropy(text)
        features.append(entropy)
        
        # Character distribution features
        num_digits = sum(c.isdigit() for c in text)
        num_alpha = sum(c.isalpha() for c in text)
        num_special = sum(not c.isalnum() and not c.isspace() for c in text)
        
        total_chars = len(text) if text else 1
        features.extend([
            num_digits / total_chars,
            num_alpha / total_chars,
            num_special / total_chars
        ])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_edge_features(self, cpg: CodePropertyGraph, edge_attr: List[str]) -> torch.Tensor:
        """Extract features for edges."""
        
        edge_features = []
        
        for edge_type_str in edge_attr:
            edge_type_id = self.edge_type_to_id.get(edge_type_str, self.edge_type_to_id['unknown'])
            edge_embedding = self.edge_type_embedding(torch.tensor(edge_type_id))
            edge_features.append(edge_embedding)
        
        if edge_features:
            return torch.stack(edge_features)
        else:
            return torch.empty((0, self.config.node_embedding_dim // 8))
    
    def _extract_global_features(self, cpg: CodePropertyGraph) -> torch.Tensor:
        """Extract package-level global features."""
        
        features = []
        
        # Graph size features
        features.extend([
            cpg.total_nodes / 1000.0,  # Normalized node count
            cpg.total_edges / 5000.0,  # Normalized edge count
            cpg.num_files / 100.0      # Normalized file count
        ])
        
        # Edge type distribution
        total_edges = sum(cpg.edge_types.values()) if cpg.edge_types else 1
        for edge_type in self.edge_type_to_id.keys():
            count = cpg.edge_types.get(edge_type, 0)
            features.append(count / total_edges)
        
        # API diversity
        features.append(len(cpg.api_calls) / len(self.config.risky_apis))
        
        # Ecosystem encoding (one-hot)
        ecosystem_features = [0.0, 0.0]  # [npm, pypi]
        if cpg.ecosystem == "npm":
            ecosystem_features[0] = 1.0
        elif cpg.ecosystem == "pypi":
            ecosystem_features[1] = 1.0
        features.extend(ecosystem_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _get_node_type_id(self, node_type: str) -> int:
        """Get or create ID for node type."""
        if node_type not in self.node_type_to_id:
            if len(self.node_type_to_id) < self.config.node_type_vocab_size - 1:
                self.node_type_to_id[node_type] = len(self.node_type_to_id)
            else:
                return self.config.node_type_vocab_size - 1  # Use last ID for unknown
        
        return self.node_type_to_id[node_type]
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
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
            entropy -= prob * math.log2(prob)
        
        # Normalize entropy (max is log2(unique_chars))
        max_entropy = math.log2(len(char_counts)) if char_counts else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def forward(self, cpg: CodePropertyGraph) -> Data:
        """Forward pass for feature extraction."""
        return self.extract_features(cpg)