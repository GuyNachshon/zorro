"""
Unified benchmark framework for ICN vs SOTA malicious package detection.
Supports ICN, HuggingFace models, OpenRouter LLMs, and traditional baselines.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import re

# ICN imports
from ..training.dataloader import ProcessedPackage
from ..evaluation.metrics import ICNMetrics
from ..evaluation.openrouter_client import OpenRouterClient, BenchmarkRequest, MaliciousPackagePrompts
from ..evaluation.llm_response_parser import LLMResponseParser

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSample:
    """Sample for benchmarking with standardized format."""
    package_name: str
    ecosystem: str  # npm, pypi
    sample_type: str  # benign, compromised_lib, malicious_intent
    ground_truth_label: int  # 0 = benign, 1 = malicious
    
    # Raw content for LLMs - different granularities
    raw_content: str  # Combined package content (existing approach)
    file_paths: List[str]
    individual_files: Dict[str, str] = field(default_factory=dict)  # filepath -> content mapping
    
    # Processed features for ML models
    processed_package: Optional[ProcessedPackage] = None
    
    # Metadata
    package_size_bytes: int = 0
    num_files: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result from a single model evaluation."""
    model_name: str
    sample_id: str
    ground_truth: int
    prediction: int  # 0 or 1
    confidence: float  # 0.0 to 1.0
    
    # Timing and cost
    inference_time_seconds: float
    cost_usd: Optional[float] = None
    
    # Model-specific outputs
    raw_output: str = ""
    explanation: str = ""
    malicious_indicators: List[str] = field(default_factory=list)
    
    # Error handling
    success: bool = True
    error_message: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    """Abstract base class for benchmark models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    async def predict(self, sample: BenchmarkSample) -> BenchmarkResult:
        """Make prediction on a sample."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass


class ICNBenchmarkModel(BaseModel):
    """ICN model wrapper for benchmarking."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__("ICN")
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        
    def _load_model(self):
        """Load ICN model from checkpoint."""
        if self.model is None:
            # Import ICN components
            import torch
            from ..models.icn_model import ICNModel
            
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize model with config from checkpoint
            model_config = checkpoint.get('model_config', {})
            self.model = ICNModel(**model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ ICN model loaded from {self.model_path}")
    
    async def predict(self, sample: BenchmarkSample) -> BenchmarkResult:
        """Run ICN prediction."""
        start_time = time.time()
        
        try:
            self._load_model()
            
            if not sample.processed_package:
                return BenchmarkResult(
                    model_name=self.model_name,
                    sample_id=f"{sample.ecosystem}_{sample.package_name}",
                    ground_truth=sample.ground_truth_label,
                    prediction=0,
                    confidence=0.0,
                    inference_time_seconds=time.time() - start_time,
                    success=False,
                    error_message="No processed package data available"
                )
            
            # Convert to ICN input format
            import torch
            from ..models.icn_model import ICNInput
            
            with torch.no_grad():
                icn_input = ICNInput(
                    input_ids=sample.processed_package.input_ids.unsqueeze(0).to(self.device),
                    attention_masks=sample.processed_package.attention_masks.unsqueeze(0).to(self.device),
                    phase_ids=sample.processed_package.phase_ids.unsqueeze(0).to(self.device),
                    api_features=sample.processed_package.api_features.unsqueeze(0).to(self.device),
                    ast_features=sample.processed_package.ast_features.unsqueeze(0).to(self.device),
                    manifest_embeddings=sample.processed_package.manifest_embedding.unsqueeze(0).to(self.device) if sample.processed_package.manifest_embedding is not None else None
                )
                
                # Run ICN forward pass
                output = self.model(icn_input)
                
                # Extract predictions
                final_score = torch.sigmoid(output.final_classification_scores[0]).item()
                prediction = 1 if final_score > 0.5 else 0
                
                # Generate explanation from ICN outputs
                explanation = self._generate_icn_explanation(output, sample.processed_package)
                
                return BenchmarkResult(
                    model_name=self.model_name,
                    sample_id=f"{sample.ecosystem}_{sample.package_name}",
                    ground_truth=sample.ground_truth_label,
                    prediction=prediction,
                    confidence=final_score,
                    inference_time_seconds=time.time() - start_time,
                    explanation=explanation,
                    raw_output=f"final_score={final_score:.4f}",
                    success=True
                )
                
        except Exception as e:
            return BenchmarkResult(
                model_name=self.model_name,
                sample_id=f"{sample.ecosystem}_{sample.package_name}",
                ground_truth=sample.ground_truth_label,
                prediction=0,
                confidence=0.0,
                inference_time_seconds=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _generate_icn_explanation(self, output, processed_package) -> str:
        """Generate human-readable explanation from ICN outputs."""
        try:
            # Extract key information from ICN output
            convergence_info = f"Converged in {len(output.convergence_history)} iterations"
            
            # Check if it's divergence or plausibility based
            if hasattr(output, 'divergence_scores') and output.divergence_scores is not None:
                max_divergence = output.divergence_scores.max().item()
                explanation = f"Divergence-based detection (max divergence: {max_divergence:.3f}). {convergence_info}."
            elif hasattr(output, 'plausibility_scores') and output.plausibility_scores is not None:
                plausibility = output.plausibility_scores.mean().item()
                explanation = f"Plausibility-based detection (plausibility: {plausibility:.3f}). {convergence_info}."
            else:
                explanation = f"ICN detection based on intent convergence analysis. {convergence_info}."
            
            return explanation
            
        except Exception:
            return "ICN-based malware detection"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ICN model information."""
        return {
            "model_type": "ICN",
            "model_path": str(self.model_path),
            "device": self.device,
            "supports_explanations": True,
            "inference_method": "local"
        }


class HuggingFaceModel(BaseModel):
    """HuggingFace model wrapper for benchmarking."""

    def __init__(self, model_name: str, model_id: str, device: str = "cuda",
                 base_model_id: str = None, adapter_id: str = None, use_peft: bool = False):
        super().__init__(model_name)
        self.model_id = model_id
        self.base_model_id = base_model_id  # For PEFT models
        self.adapter_id = adapter_id        # For PEFT adapters
        self.use_peft = use_peft
        self.device = device
        self.model = None
        self.tokenizer = None
    
    def _load_model(self):
        """Load HuggingFace model (with PEFT support)."""
        if self.model is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            if self.use_peft:
                # Load PEFT model
                try:
                    from peft import PeftModel

                    # Load base model and tokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
                    base_model = AutoModelForSequenceClassification.from_pretrained(self.base_model_id)

                    # Load PEFT adapter
                    self.model = PeftModel.from_pretrained(base_model, self.adapter_id)
                    self.model.to(self.device)
                    self.model.eval()

                    logger.info(f"✅ PEFT model loaded: base={self.base_model_id}, adapter={self.adapter_id}")

                except ImportError:
                    logger.error("PEFT library not available. Install with: pip install peft")
                    raise
                except Exception as e:
                    logger.error(f"Failed to load PEFT model: {e}")
                    raise
            else:
                # Load regular HuggingFace model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
                self.model.to(self.device)
                self.model.eval()

                logger.info(f"✅ HuggingFace model loaded: {self.model_id}")
    
    async def predict(self, sample: BenchmarkSample) -> BenchmarkResult:
        """Run HuggingFace model prediction."""
        start_time = time.time()
        
        try:
            self._load_model()
            
            # Prepare text input (truncate to fit model context)
            text_input = sample.raw_content[:8000]  # Most BERT models have ~512 token limit
            
            # Tokenize
            inputs = self.tokenizer(
                text_input,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            import torch
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get prediction and confidence
                if logits.size(-1) == 2:  # Binary classification
                    probabilities = torch.softmax(logits, dim=-1)
                    confidence = probabilities[0, 1].item()  # Probability of malicious class
                    prediction = 1 if confidence > 0.5 else 0
                else:  # Single output
                    confidence = torch.sigmoid(logits[0, 0]).item()
                    prediction = 1 if confidence > 0.5 else 0
            
            return BenchmarkResult(
                model_name=self.model_name,
                sample_id=f"{sample.ecosystem}_{sample.package_name}",
                ground_truth=sample.ground_truth_label,
                prediction=prediction,
                confidence=confidence,
                inference_time_seconds=time.time() - start_time,
                raw_output=f"logits={logits.tolist()}",
                success=True
            )
            
        except Exception as e:
            return BenchmarkResult(
                model_name=self.model_name,
                sample_id=f"{sample.ecosystem}_{sample.package_name}",
                ground_truth=sample.ground_truth_label,
                prediction=0,
                confidence=0.0,
                inference_time_seconds=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get HuggingFace model information."""
        return {
            "model_type": "HuggingFace",
            "model_id": self.model_id,
            "device": self.device,
            "supports_explanations": False,
            "inference_method": "local"
        }


class OpenRouterModel(BaseModel):
    """OpenRouter LLM wrapper for benchmarking."""
    
    def __init__(self, model_name: str, openrouter_model_id: str, prompt_type: str = "zero_shot",
                 granularity: str = "package"):
        super().__init__(f"{model_name}_{granularity}")
        self.original_model_name = model_name
        self.openrouter_model_id = openrouter_model_id
        self.prompt_type = prompt_type  # zero_shot, few_shot, reasoning
        self.granularity = granularity  # "package" or "file_by_file"
        self.examples = []  # For few-shot prompting
        self.response_parser = LLMResponseParser()  # Parser for LLM responses
    
    def set_few_shot_examples(self, examples: List[Dict[str, Any]]):
        """Set examples for few-shot prompting."""
        self.examples = examples
        self.prompt_type = "few_shot"
    
    async def predict(self, sample: BenchmarkSample, openrouter_client: Optional[OpenRouterClient] = None) -> BenchmarkResult:
        """Run LLM prediction via OpenRouter."""
        if self.granularity == "file_by_file":
            return await self._predict_file_by_file(sample, openrouter_client)
        else:
            return await self._predict_package_level(sample, openrouter_client)
    
    async def _predict_package_level(self, sample: BenchmarkSample, openrouter_client: Optional[OpenRouterClient]) -> BenchmarkResult:
        """Standard package-level prediction."""
        start_time = time.time()
        
        try:
            if not openrouter_client:
                return BenchmarkResult(
                    model_name=self.model_name,
                    sample_id=f"{sample.ecosystem}_{sample.package_name}",
                    ground_truth=sample.ground_truth_label,
                    prediction=0,
                    confidence=0.5,
                    inference_time_seconds=time.time() - start_time,
                    success=False,
                    error_message="OpenRouter client not provided"
                )
            
            # Generate appropriate prompt
            if self.prompt_type == "few_shot" and self.examples:
                prompt = MaliciousPackagePrompts.few_shot_prompt(sample.raw_content, self.examples)
            elif self.prompt_type == "reasoning":
                prompt = MaliciousPackagePrompts.reasoning_prompt(sample.raw_content)
            else:
                prompt = MaliciousPackagePrompts.zero_shot_prompt(sample.raw_content)
            
            # Make request through OpenRouter
            request = BenchmarkRequest(
                prompt=prompt,
                model_name=self.openrouter_model_id,
                temperature=0.0,
                max_tokens=1000,
                metadata={"sample_id": f"{sample.ecosystem}_{sample.package_name}"}
            )
            
            # Get LLM response
            llm_response = await openrouter_client.generate_response(request)
            
            if not llm_response.success:
                return BenchmarkResult(
                    model_name=self.model_name,
                    sample_id=f"{sample.ecosystem}_{sample.package_name}",
                    ground_truth=sample.ground_truth_label,
                    prediction=0,
                    confidence=0.0,
                    inference_time_seconds=time.time() - start_time,
                    cost_usd=llm_response.cost_usd,
                    success=False,
                    error_message=llm_response.error_message or "LLM request failed",
                    raw_output=llm_response.response_text
                )
            
            # Parse LLM response to extract structured prediction
            parsed = self.response_parser.parse_response(llm_response.response_text, self.model_name)
            
            return BenchmarkResult(
                model_name=self.model_name,
                sample_id=f"{sample.ecosystem}_{sample.package_name}",
                ground_truth=sample.ground_truth_label,
                prediction=1 if parsed.is_malicious else 0,
                confidence=parsed.confidence,
                inference_time_seconds=time.time() - start_time,
                cost_usd=llm_response.cost_usd,
                raw_output=llm_response.response_text,
                explanation=parsed.reasoning,
                malicious_indicators=parsed.malicious_indicators,
                success=True,
                metadata={
                    "parse_method": parsed.parse_method,
                    "parse_success": parsed.parse_success,
                    "prompt_tokens": llm_response.prompt_tokens,
                    "completion_tokens": llm_response.completion_tokens,
                    "granularity": "package"
                }
            )
            
        except Exception as e:
            return BenchmarkResult(
                model_name=self.model_name,
                sample_id=f"{sample.ecosystem}_{sample.package_name}",
                ground_truth=sample.ground_truth_label,
                prediction=0,
                confidence=0.0,
                inference_time_seconds=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    async def _predict_file_by_file(self, sample: BenchmarkSample, openrouter_client: Optional[OpenRouterClient]) -> BenchmarkResult:
        """File-by-file prediction with aggregation."""
        start_time = time.time()
        
        try:
            if not openrouter_client:
                return BenchmarkResult(
                    model_name=self.model_name,
                    sample_id=f"{sample.ecosystem}_{sample.package_name}",
                    ground_truth=sample.ground_truth_label,
                    prediction=0,
                    confidence=0.5,
                    inference_time_seconds=time.time() - start_time,
                    success=False,
                    error_message="OpenRouter client not provided"
                )
            
            # Check if we have individual files
            if not sample.individual_files:
                # Fallback to package-level analysis
                logger.warning(f"No individual files for {sample.package_name}, falling back to package-level analysis")
                return await self._predict_package_level(sample, openrouter_client)
            
            # Analyze each file individually with early stopping
            file_analyses = []
            total_cost = 0.0
            total_tokens = 0
            found_malicious = False

            # Limit files to avoid excessive API calls (max 8 files)
            files_to_analyze = list(sample.individual_files.items())[:8]

            for file_path, file_content in files_to_analyze:
                if not file_content.strip():
                    continue

                # Create file-specific prompt
                prompt = MaliciousPackagePrompts.file_by_file_prompt(file_path, file_content)

                # Create request
                request = BenchmarkRequest(
                    prompt=prompt,
                    model_name=self.openrouter_model_id,
                    temperature=0.0,
                    max_tokens=800,  # Smaller for individual files
                    metadata={"file_path": file_path, "sample_id": f"{sample.ecosystem}_{sample.package_name}"}
                )

                # Get LLM response
                llm_response = await openrouter_client.generate_response(request)

                if llm_response.success:
                    # Parse individual file response
                    parsed = self.response_parser.parse_response(llm_response.response_text, self.model_name)

                    file_analysis = {
                        "file_path": file_path,
                        "is_malicious": parsed.is_malicious,
                        "confidence": parsed.confidence,
                        "reasoning": parsed.reasoning,
                        "malicious_indicators": parsed.malicious_indicators,
                        "risk_level": "high" if parsed.confidence > 0.8 and parsed.is_malicious else
                                     "medium" if parsed.confidence > 0.5 and parsed.is_malicious else "low"
                    }

                    file_analyses.append(file_analysis)
                    total_cost += llm_response.cost_usd
                    total_tokens += llm_response.total_tokens

                    # Early stopping: if we found malicious content with high confidence, stop
                    if parsed.is_malicious and parsed.confidence >= 0.8:
                        found_malicious = True
                        logger.info(f"🛑 Early stopping: Found malicious content in {file_path} with confidence {parsed.confidence:.2f}")
                        break
                    
                    # Small delay to respect rate limits
                    await asyncio.sleep(0.1)
            
            if not file_analyses:
                return BenchmarkResult(
                    model_name=self.model_name,
                    sample_id=f"{sample.ecosystem}_{sample.package_name}",
                    ground_truth=sample.ground_truth_label,
                    prediction=0,
                    confidence=0.0,
                    inference_time_seconds=time.time() - start_time,
                    success=False,
                    error_message="No files could be analyzed"
                )
            
            # Aggregate the results using simple statistical method
            aggregated_result = self._simple_aggregation(
                sample, file_analyses, total_cost, total_tokens, time.time() - start_time,
                early_stopped=found_malicious, total_files_available=len(files_to_analyze)
            )

            return aggregated_result
            
        except Exception as e:
            return BenchmarkResult(
                model_name=self.model_name,
                sample_id=f"{sample.ecosystem}_{sample.package_name}",
                ground_truth=sample.ground_truth_label,
                prediction=0,
                confidence=0.0,
                inference_time_seconds=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _simple_aggregation(self, sample: BenchmarkSample, file_analyses: List[Dict],
                           total_cost: float, total_tokens: int, total_time: float,
                           early_stopped: bool = False, total_files_available: int = None) -> BenchmarkResult:
        """Simple statistical aggregation of file analyses."""
        
        malicious_files = [f for f in file_analyses if f.get("is_malicious", False)]
        malicious_count = len(malicious_files)
        total_files = len(file_analyses)
        
        # Package is malicious if any file is malicious with high confidence
        # or if multiple files are malicious with medium confidence
        high_confidence_malicious = [f for f in malicious_files if f.get("confidence", 0) > 0.8]
        medium_confidence_malicious = [f for f in malicious_files if f.get("confidence", 0) > 0.5]
        
        is_malicious = (len(high_confidence_malicious) > 0) or (len(medium_confidence_malicious) > 1)
        
        if is_malicious:
            # Confidence is the maximum confidence of malicious files, adjusted by proportion
            max_confidence = max([f.get("confidence", 0) for f in malicious_files])
            proportion_malicious = malicious_count / total_files
            confidence = max_confidence * (0.7 + 0.3 * proportion_malicious)  # Adjust by proportion
        else:
            # Confidence in benign classification
            benign_files = [f for f in file_analyses if not f.get("is_malicious", False)]
            if benign_files:
                avg_benign_confidence = np.mean([f.get("confidence", 0.5) for f in benign_files])
                confidence = min(0.9, avg_benign_confidence)
            else:
                confidence = 0.5
        
        # Create explanation
        if early_stopped:
            explanation = f"File-by-file (early stop): {malicious_count}/{total_files} files analyzed, stopped at first high-confidence malicious file."
        else:
            files_checked = f"{total_files}/{total_files_available}" if total_files_available else str(total_files)
            explanation = f"File-by-file analysis: {malicious_count}/{files_checked} files flagged as malicious."

        if malicious_files:
            top_risk_files = sorted(malicious_files, key=lambda x: x.get('confidence', 0), reverse=True)[:3]
            explanation += f" Highest risk files: {[f['file_path'] for f in top_risk_files]}"
        
        # Aggregate malicious indicators
        all_indicators = []
        for analysis in malicious_files:
            all_indicators.extend(analysis.get("malicious_indicators", []))
        unique_indicators = list(set(all_indicators))
        
        return BenchmarkResult(
            model_name=self.model_name,
            sample_id=f"{sample.ecosystem}_{sample.package_name}",
            ground_truth=sample.ground_truth_label,
            prediction=1 if is_malicious else 0,
            confidence=min(1.0, max(0.0, confidence)),
            inference_time_seconds=total_time,
            cost_usd=total_cost,
            explanation=explanation,
            malicious_indicators=unique_indicators,
            success=True,
            metadata={
                "granularity": "file_by_file",
                "files_analyzed": total_files,
                "malicious_files": malicious_count,
                "aggregation_method": "simple_statistical",
                "total_tokens": total_tokens,
                "individual_file_analyses": file_analyses[:3]  # Store first 3 for debugging
            }
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenRouter model information."""
        return {
            "model_type": "OpenRouter_LLM",
            "openrouter_model_id": self.openrouter_model_id,
            "prompt_type": self.prompt_type,
            "granularity": self.granularity,
            "supports_explanations": True,
            "inference_method": "api"
        }


class BaselineModel(BaseModel):
    """Traditional baseline models (heuristics, classical ML)."""
    
    def __init__(self, model_name: str, model_type: str):
        super().__init__(model_name)
        self.model_type = model_type  # heuristic, random_forest, svm
    
    async def predict(self, sample: BenchmarkSample) -> BenchmarkResult:
        """Run baseline prediction."""
        start_time = time.time()
        
        try:
            if self.model_type == "heuristic":
                prediction, confidence = self._heuristic_detection(sample)
            elif self.model_type == "random":
                # Random baseline for comparison
                prediction = np.random.choice([0, 1])
                confidence = np.random.uniform(0.4, 0.6)
            else:
                # Placeholder for other baseline methods
                prediction, confidence = 0, 0.5
            
            return BenchmarkResult(
                model_name=self.model_name,
                sample_id=f"{sample.ecosystem}_{sample.package_name}",
                ground_truth=sample.ground_truth_label,
                prediction=prediction,
                confidence=confidence,
                inference_time_seconds=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            return BenchmarkResult(
                model_name=self.model_name,
                sample_id=f"{sample.ecosystem}_{sample.package_name}",
                ground_truth=sample.ground_truth_label,
                prediction=0,
                confidence=0.0,
                inference_time_seconds=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _heuristic_detection(self, sample: BenchmarkSample) -> Tuple[int, float]:
        """Simple heuristic-based detection."""
        content = sample.raw_content.lower()
        
        # Suspicious patterns
        suspicious_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'base64\.decode',
            r'urllib\.request',
            r'subprocess\.call',
            r'os\.system',
            r'shell=true',
            r'crypto',
            r'encrypt',
            r'decode',
            r'obfuscat'
        ]
        
        # Count suspicious patterns
        matches = 0
        for pattern in suspicious_patterns:
            matches += len(re.findall(pattern, content))
        
        # Simple scoring
        if matches >= 5:
            return 1, 0.9
        elif matches >= 3:
            return 1, 0.7
        elif matches >= 1:
            return 1, 0.6
        else:
            return 0, 0.3
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get baseline model information."""
        return {
            "model_type": "Baseline",
            "baseline_type": self.model_type,
            "supports_explanations": False,
            "inference_method": "local"
        }


class BenchmarkSuite:
    """Main benchmark orchestration class."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, BaseModel] = {}
        self.samples: List[BenchmarkSample] = []
        self.results: List[BenchmarkResult] = []
        
    def register_model(self, model: BaseModel):
        """Register a model for benchmarking."""
        self.models[model.model_name] = model
        logger.info(f"📝 Registered model: {model.model_name}")
    
    def load_samples(self, samples: List[BenchmarkSample]):
        """Load samples for benchmarking."""
        self.samples = samples
        logger.info(f"📊 Loaded {len(samples)} benchmark samples")
        
        # Print sample distribution
        distribution = {}
        for sample in samples:
            key = f"{sample.sample_type}"
            distribution[key] = distribution.get(key, 0) + 1
        
        logger.info("Sample distribution:")
        for sample_type, count in distribution.items():
            logger.info(f"   {sample_type}: {count}")
    
    async def run_benchmark(self, max_concurrent: int = 3) -> pd.DataFrame:
        """Run complete benchmark across all models and samples."""
        logger.info(f"🚀 Starting benchmark: {len(self.models)} models × {len(self.samples)} samples")
        
        all_results = []
        
        for model_name, model in self.models.items():
            logger.info(f"🤖 Running benchmark for {model_name}...")
            
            # Create tasks for this model
            tasks = [model.predict(sample) for sample in self.samples]
            
            # Run with concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def limited_task(task):
                async with semaphore:
                    return await task
            
            limited_tasks = [limited_task(task) for task in tasks]
            model_results = await asyncio.gather(*limited_tasks)
            
            # Track success rate
            successful = sum(1 for r in model_results if r.success)
            logger.info(f"   ✅ {successful}/{len(model_results)} predictions successful")
            
            all_results.extend(model_results)
        
        self.results = all_results
        
        # Convert to DataFrame for analysis
        return self._results_to_dataframe()
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = []
        for result in self.results:
            data.append({
                'model_name': result.model_name,
                'sample_id': result.sample_id,
                'ground_truth': result.ground_truth,
                'prediction': result.prediction,
                'confidence': result.confidence,
                'inference_time': result.inference_time_seconds,
                'cost_usd': result.cost_usd,
                'success': result.success,
                'error_message': result.error_message
            })
        
        return pd.DataFrame(data)
    
    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """Compute performance metrics for all models."""
        df = self._results_to_dataframe()
        
        # Filter successful predictions only
        successful_df = df[df['success'] == True]
        
        metrics_computer = ICNMetrics()
        all_metrics = {}
        
        for model_name in successful_df['model_name'].unique():
            model_df = successful_df[successful_df['model_name'] == model_name]
            
            if len(model_df) == 0:
                continue
            
            # Compute metrics
            predictions = model_df['confidence'].values
            labels = model_df['ground_truth'].values
            
            metrics = metrics_computer.compute_metrics(predictions, labels, prefix=model_name)
            
            # Add timing and cost metrics
            metrics[f"{model_name}_avg_inference_time"] = model_df['inference_time'].mean()
            metrics[f"{model_name}_total_cost_usd"] = model_df['cost_usd'].sum() if 'cost_usd' in model_df else 0.0
            metrics[f"{model_name}_success_rate"] = len(model_df) / len(df[df['model_name'] == model_name])
            
            all_metrics[model_name] = metrics
        
        return all_metrics
    
    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        metrics = self.compute_metrics()
        
        report = f"""
# ICN Malicious Package Detection Benchmark Report

Generated: {datetime.now().isoformat()}

## Overview
- Models tested: {len(self.models)}
- Samples evaluated: {len(self.samples)}
- Total predictions: {len(self.results)}

## Model Performance Summary

| Model | F1 Score | Precision | Recall | ROC-AUC | Avg Time (s) | Success Rate |
|-------|----------|-----------|---------|---------|--------------|--------------|
"""
        
        for model_name, model_metrics in metrics.items():
            f1 = model_metrics.get(f"{model_name}_f1", 0.0)
            precision = model_metrics.get(f"{model_name}_precision", 0.0)
            recall = model_metrics.get(f"{model_name}_recall", 0.0)
            roc_auc = model_metrics.get(f"{model_name}_roc_auc", 0.0)
            avg_time = model_metrics.get(f"{model_name}_avg_inference_time", 0.0)
            success_rate = model_metrics.get(f"{model_name}_success_rate", 0.0)
            
            report += f"| {model_name} | {f1:.3f} | {precision:.3f} | {recall:.3f} | {roc_auc:.3f} | {avg_time:.3f} | {success_rate:.3f} |\n"
        
        return report
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save detailed results to file."""
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "models": {name: model.get_model_info() for name, model in self.models.items()},
            "samples_count": len(self.samples),
            "results": [
                {
                    "model_name": r.model_name,
                    "sample_id": r.sample_id,
                    "ground_truth": r.ground_truth,
                    "prediction": r.prediction,
                    "confidence": r.confidence,
                    "inference_time_seconds": r.inference_time_seconds,
                    "cost_usd": r.cost_usd,
                    "success": r.success,
                    "error_message": r.error_message,
                    "explanation": r.explanation,
                    "raw_output": r.raw_output
                }
                for r in self.results
            ],
            "metrics": self.compute_metrics()
        }
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    logger.basicConfig(level=logging.INFO)
    
    print("🧪 ICN Benchmark Framework initialized")
    print("Ready for comprehensive model comparison!")