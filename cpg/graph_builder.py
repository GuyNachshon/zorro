"""
Code Property Graph builder for AST, Control Flow, and Data Flow analysis.
"""

import ast
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union

import networkx as nx
import numpy as np
import tree_sitter
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript

from .config import CPGConfig

logger = logging.getLogger(__name__)


@dataclass
class CodePropertyGraph:
    """Represents a Code Property Graph for a package."""
    
    package_name: str
    ecosystem: str  # npm, pypi
    
    # NetworkX graph
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    
    # Node mappings
    node_to_code: Dict[int, str] = field(default_factory=dict)  # node_id -> code snippet
    node_to_type: Dict[int, str] = field(default_factory=dict)  # node_id -> AST type
    node_to_file: Dict[int, str] = field(default_factory=dict)  # node_id -> file path
    
    # Edge type counts
    edge_types: Dict[str, int] = field(default_factory=dict)
    
    # Metadata
    num_files: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    api_calls: Set[str] = field(default_factory=set)
    
    def add_node(self, node_id: int, **attrs):
        """Add a node to the graph."""
        self.graph.add_node(node_id, **attrs)
        self.total_nodes += 1
    
    def add_edge(self, source: int, target: int, edge_type: str, **attrs):
        """Add an edge to the graph."""
        self.graph.add_edge(source, target, edge_type=edge_type, **attrs)
        self.edge_types[edge_type] = self.edge_types.get(edge_type, 0) + 1
        self.total_edges += 1
    
    def to_pyg_data(self):
        """Convert to PyTorch Geometric Data object."""
        import torch
        from torch_geometric.data import Data
        
        # Get node features (will be populated by feature extractor)
        node_mapping = {node: i for i, node in enumerate(self.graph.nodes())}
        
        # Convert edges
        edge_index = []
        edge_attr = []
        
        for source, target, data in self.graph.edges(data=True):
            edge_index.append([node_mapping[source], node_mapping[target]])
            edge_attr.append(data.get('edge_type', 'unknown'))
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create PyG data object
        data = Data(
            edge_index=edge_index,
            num_nodes=len(self.graph.nodes()),
            node_mapping=node_mapping,
            edge_attr=edge_attr,
            package_name=self.package_name,
            ecosystem=self.ecosystem
        )
        
        return data


class CPGBuilder:
    """Builds Code Property Graphs from source code."""
    
    def __init__(self, config: CPGConfig):
        self.config = config
        self.node_counter = 0
        
        # Initialize Tree-sitter parsers
        self.python_parser = Parser()
        self.js_parser = Parser()
        
        # Build languages
        PY_LANGUAGE = Language(tspython.language())
        JS_LANGUAGE = Language(tsjavascript.language())
        
        self.python_parser.language = PY_LANGUAGE
        self.js_parser.language = JS_LANGUAGE
        
        # Risky API patterns
        self.risky_apis = set(config.risky_apis)
    
    def build_package_graph(self, 
                           package_name: str,
                           ecosystem: str,
                           file_contents: Dict[str, str]) -> CodePropertyGraph:
        """Build a complete CPG for a package."""
        
        cpg = CodePropertyGraph(package_name=package_name, ecosystem=ecosystem)
        cpg.num_files = len(file_contents)
        
        # Build graph for each file
        file_graphs = []
        for file_path, content in file_contents.items():
            if not content or not content.strip():
                continue
                
            file_graph = self._build_file_graph(file_path, content, ecosystem)
            if file_graph:
                file_graphs.append((file_path, file_graph))
        
        # Combine file graphs
        if file_graphs:
            cpg = self._combine_file_graphs(cpg, file_graphs)
        
        # Add metadata node
        self._add_metadata_node(cpg)
        
        # Prune if graph is too large
        if cpg.total_nodes > self.config.max_nodes_per_graph:
            cpg = self._prune_graph(cpg)
        
        return cpg
    
    def _build_file_graph(self, file_path: str, content: str, ecosystem: str) -> Optional[nx.DiGraph]:
        """Build graph for a single file."""
        
        graph = nx.DiGraph()
        
        try:
            # Determine file type and parser
            if ecosystem == "pypi" or file_path.endswith('.py'):
                parser = self.python_parser
                tree = parser.parse(bytes(content, "utf8"))
                self._build_python_graph(graph, tree.root_node, content, file_path)
            elif ecosystem == "npm" or file_path.endswith(('.js', '.ts')):
                parser = self.js_parser
                tree = parser.parse(bytes(content, "utf8"))
                self._build_javascript_graph(graph, tree.root_node, content, file_path)
            else:
                # Try Python AST as fallback
                try:
                    tree = ast.parse(content)
                    self._build_python_ast_graph(graph, tree, content, file_path)
                except:
                    logger.warning(f"Could not parse file: {file_path}")
                    return None
            
            # Add control flow edges if enabled
            if self.config.include_cfg_edges:
                self._add_control_flow_edges(graph)
            
            # Add data flow edges if enabled
            if self.config.include_dfg_edges:
                self._add_data_flow_edges(graph)
            
            return graph
            
        except Exception as e:
            logger.error(f"Error building graph for {file_path}: {e}")
            return None
    
    def _build_python_graph(self, graph: nx.DiGraph, node, content: str, file_path: str):
        """Build graph from Tree-sitter Python AST."""
        
        node_id = self._get_next_node_id()
        
        # Extract node information
        node_type = node.type
        start_byte = node.start_byte
        end_byte = node.end_byte
        node_text = content[start_byte:end_byte]
        
        # Add node to graph
        graph.add_node(
            node_id,
            type=node_type,
            text=node_text[:100],  # Truncate long text
            file=file_path,
            start_byte=start_byte,
            end_byte=end_byte
        )
        
        # Check for API calls
        if node_type in ["call", "attribute"]:
            self._check_api_call(node_text)
        
        # Process children
        parent_id = node_id
        for child in node.children:
            child_id = self._build_python_graph(graph, child, content, file_path)
            if child_id is not None:
                graph.add_edge(parent_id, child_id, edge_type="ast")
        
        return node_id
    
    def _build_javascript_graph(self, graph: nx.DiGraph, node, content: str, file_path: str):
        """Build graph from Tree-sitter JavaScript AST."""
        
        node_id = self._get_next_node_id()
        
        # Extract node information
        node_type = node.type
        start_byte = node.start_byte
        end_byte = node.end_byte
        node_text = content[start_byte:end_byte]
        
        # Add node to graph
        graph.add_node(
            node_id,
            type=node_type,
            text=node_text[:100],
            file=file_path,
            start_byte=start_byte,
            end_byte=end_byte
        )
        
        # Check for API calls
        if node_type in ["call_expression", "member_expression"]:
            self._check_api_call(node_text)
        
        # Process children
        parent_id = node_id
        for child in node.children:
            child_id = self._build_javascript_graph(graph, child, content, file_path)
            if child_id is not None:
                graph.add_edge(parent_id, child_id, edge_type="ast")
        
        return node_id
    
    def _build_python_ast_graph(self, graph: nx.DiGraph, tree, content: str, file_path: str):
        """Build graph from Python AST (fallback method)."""
        
        def visit_node(node, parent_id=None):
            node_id = self._get_next_node_id()
            node_type = node.__class__.__name__
            
            # Get node text (approximation)
            node_text = ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
            
            graph.add_node(
                node_id,
                type=node_type,
                text=node_text[:100],
                file=file_path
            )
            
            # Check for API calls
            if isinstance(node, ast.Call):
                self._check_api_call(node_text)
            
            if parent_id is not None:
                graph.add_edge(parent_id, node_id, edge_type="ast")
            
            # Visit children
            for child in ast.iter_child_nodes(node):
                visit_node(child, node_id)
            
            return node_id
        
        visit_node(tree)
    
    def _add_control_flow_edges(self, graph: nx.DiGraph):
        """Add control flow edges to the graph."""
        
        # Find control flow nodes
        control_nodes = []
        for node_id, data in graph.nodes(data=True):
            node_type = data.get('type', '')
            if node_type in ['if_statement', 'while_statement', 'for_statement', 
                            'try_statement', 'If', 'While', 'For', 'Try']:
                control_nodes.append(node_id)
        
        # Add CFG edges between control nodes
        for i, source in enumerate(control_nodes[:-1]):
            target = control_nodes[i + 1]
            if not graph.has_edge(source, target):
                graph.add_edge(source, target, edge_type="cfg")
    
    def _add_data_flow_edges(self, graph: nx.DiGraph):
        """Add data flow edges to the graph."""
        
        # Track variable definitions and uses
        definitions = {}  # var_name -> node_id
        
        for node_id, data in graph.nodes(data=True):
            node_type = data.get('type', '')
            node_text = data.get('text', '')
            
            # Check for assignments (definitions)
            if node_type in ['assignment', 'Assign', '=']:
                # Extract variable name (simplified)
                if '=' in node_text:
                    var_name = node_text.split('=')[0].strip()
                    definitions[var_name] = node_id
            
            # Check for variable uses
            for var_name, def_node in definitions.items():
                if var_name in node_text and node_id != def_node:
                    if not graph.has_edge(def_node, node_id):
                        graph.add_edge(def_node, node_id, edge_type="dfg")
    
    def _combine_file_graphs(self, cpg: CodePropertyGraph, 
                           file_graphs: List[Tuple[str, nx.DiGraph]]) -> CodePropertyGraph:
        """Combine individual file graphs into package graph."""
        
        # Add all file graphs to the main CPG
        for file_path, file_graph in file_graphs:
            # Add nodes
            for node_id, data in file_graph.nodes(data=True):
                cpg.add_node(node_id, **data)
                cpg.node_to_code[node_id] = data.get('text', '')
                cpg.node_to_type[node_id] = data.get('type', '')
                cpg.node_to_file[node_id] = file_path
            
            # Add edges
            for source, target, data in file_graph.edges(data=True):
                edge_type = data.pop('edge_type', 'unknown')  # Remove from data dict
                cpg.add_edge(source, target, edge_type, **data)
        
        # Add inter-file edges (imports, exports)
        self._add_inter_file_edges(cpg, file_graphs)
        
        return cpg
    
    def _add_inter_file_edges(self, cpg: CodePropertyGraph, 
                             file_graphs: List[Tuple[str, nx.DiGraph]]):
        """Add edges between files (imports/exports)."""
        
        # Find import/export nodes
        import_nodes = []
        export_nodes = []
        
        for node_id, data in cpg.graph.nodes(data=True):
            node_type = data.get('type', '')
            if node_type in ['import_statement', 'Import', 'ImportFrom']:
                import_nodes.append(node_id)
            elif node_type in ['export_statement', 'Export', 'Return']:
                export_nodes.append(node_id)
        
        # Connect imports to exports (simplified)
        for imp_node in import_nodes:
            for exp_node in export_nodes:
                # Check if they're from different files
                if cpg.node_to_file.get(imp_node) != cpg.node_to_file.get(exp_node):
                    cpg.add_edge(imp_node, exp_node, "inter_file")
    
    def _add_metadata_node(self, cpg: CodePropertyGraph):
        """Add a special metadata node connected to all subgraphs."""
        
        metadata_node_id = self._get_next_node_id()
        
        cpg.add_node(
            metadata_node_id,
            type="metadata",
            text=f"Package: {cpg.package_name}",
            file="__metadata__",
            num_files=cpg.num_files,
            ecosystem=cpg.ecosystem,
            api_calls=list(cpg.api_calls)
        )
        
        # Connect metadata node to root nodes of each file
        root_nodes = []
        for node_id, data in cpg.graph.nodes(data=True):
            # Find nodes with no incoming edges (likely file roots)
            if cpg.graph.in_degree(node_id) == 0 and node_id != metadata_node_id:
                root_nodes.append(node_id)
        
        for root_node in root_nodes[:10]:  # Limit connections
            cpg.add_edge(metadata_node_id, root_node, "metadata_link")
    
    def _prune_graph(self, cpg: CodePropertyGraph) -> CodePropertyGraph:
        """Prune graph if it exceeds size limits."""
        
        logger.warning(f"Pruning graph from {cpg.total_nodes} nodes to {self.config.max_nodes_per_graph}")
        
        # Keep most important nodes (with most connections)
        node_importance = {}
        for node in cpg.graph.nodes():
            degree = cpg.graph.degree(node)
            node_type = cpg.node_to_type.get(node, '')
            
            # Prioritize certain node types
            type_weight = 1.0
            if node_type in ['call', 'Call', 'call_expression']:
                type_weight = 2.0
            elif node_type == 'metadata':
                type_weight = 10.0
            
            node_importance[node] = degree * type_weight
        
        # Keep top-k nodes
        important_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)
        nodes_to_keep = set([n for n, _ in important_nodes[:self.config.max_nodes_per_graph]])
        
        # Create pruned graph
        pruned_graph = cpg.graph.subgraph(nodes_to_keep).copy()
        cpg.graph = pruned_graph
        cpg.total_nodes = len(pruned_graph.nodes())
        cpg.total_edges = len(pruned_graph.edges())
        
        return cpg
    
    def _check_api_call(self, text: str):
        """Check if text contains risky API calls."""
        for api in self.risky_apis:
            if api in text:
                return api
        return None
    
    def _get_next_node_id(self) -> int:
        """Get next unique node ID."""
        self.node_counter += 1
        return self.node_counter