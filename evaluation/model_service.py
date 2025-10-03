"""
Model Service Architecture - Load models once, serve many predictions.
Avoids reloading models for each evaluation loop.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
from dataclasses import dataclass
from enum import Enum

from .config import ModelConfig
from icn.evaluation.benchmark_framework import BenchmarkSample, BenchmarkResult

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Status of a model in the service."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


@dataclass
class LoadedModel:
    """Container for a loaded model with metadata."""
    config: ModelConfig
    model: Any  # The actual model instance
    tokenizer: Any = None  # For HF models
    status: ModelStatus = ModelStatus.READY
    error_message: Optional[str] = None
    load_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    prediction_count: int = 0


class ModelService:
    """
    Centralized service for loading and serving model predictions.
    Models are loaded once and reused across evaluations.
    """

    def __init__(self, device: str = "auto"):
        self.device = self._resolve_device(device)
        self.models: Dict[str, LoadedModel] = {}
        self._lock = asyncio.Lock()

        logger.info(f"🚀 ModelService initialized on device: {self.device}")

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    async def load_model(self, model_config: ModelConfig) -> bool:
        """Load a model into the service."""
        async with self._lock:
            if model_config.name in self.models:
                logger.info(f"Model {model_config.name} already loaded")
                return True

            logger.info(f"🔄 Loading model: {model_config.name} ({model_config.type})")

            # Create placeholder with loading status
            self.models[model_config.name] = LoadedModel(
                config=model_config,
                model=None,
                status=ModelStatus.LOADING
            )

            start_time = time.time()
            try:
                loaded_model = await self._load_model_by_type(model_config)
                load_time = time.time() - start_time

                # Update with loaded model
                self.models[model_config.name] = LoadedModel(
                    config=model_config,
                    model=loaded_model['model'],
                    tokenizer=loaded_model.get('tokenizer'),
                    status=ModelStatus.READY,
                    load_time_seconds=load_time,
                    memory_usage_mb=self._estimate_memory_usage(loaded_model['model'])
                )

                logger.info(f"✅ Model {model_config.name} loaded in {load_time:.1f}s")
                return True

            except Exception as e:
                error_msg = f"Failed to load {model_config.name}: {e}"
                logger.error(error_msg)

                self.models[model_config.name] = LoadedModel(
                    config=model_config,
                    model=None,
                    status=ModelStatus.ERROR,
                    error_message=error_msg
                )
                return False

    async def _load_model_by_type(self, config: ModelConfig) -> Dict[str, Any]:
        """Load model based on type."""
        import time

        if config.type == "huggingface":
            return await self._load_huggingface_model(config)
        elif config.type == "amil":
            return await self._load_amil_model(config)
        elif config.type == "cpg":
            return await self._load_cpg_model(config)
        elif config.type == "neobert":
            return await self._load_neobert_model(config)
        elif config.type == "icn":
            return await self._load_icn_model(config)
        else:
            raise ValueError(f"Unsupported model type: {config.type}")

    async def _load_huggingface_model(self, config: ModelConfig) -> Dict[str, Any]:
        """Load HuggingFace model (with PEFT support)."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if config.use_peft:
            # Load PEFT model
            from peft import PeftModel

            # Load base model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.hf_base_model_id)
            base_model = AutoModelForSequenceClassification.from_pretrained(config.hf_base_model_id)

            # Load PEFT adapter
            model = PeftModel.from_pretrained(base_model, config.hf_adapter_id)
            model.to(self.device)
            model.eval()

            logger.info(f"PEFT model loaded: base={config.hf_base_model_id}, adapter={config.hf_adapter_id}")
        else:
            # Regular HuggingFace model
            tokenizer = AutoTokenizer.from_pretrained(config.hf_model_id)
            model = AutoModelForSequenceClassification.from_pretrained(config.hf_model_id)
            model.to(self.device)
            model.eval()

            logger.info(f"HuggingFace model loaded: {config.hf_model_id}")

        return {"model": model, "tokenizer": tokenizer}

    async def _load_amil_model(self, config: ModelConfig) -> Dict[str, Any]:
        """Load AMIL model."""
        # Import AMIL components
        from amil.model import AMILModel
        from amil.config import load_config_from_json

        model_path = config.model_path or f"checkpoints/{config.name}/amil_model.pth"

        if not Path(model_path).exists():
            raise FileNotFoundError(f"AMIL model not found: {model_path}")

        # Load config and model
        config_path = Path(model_path).parent / "config.json"
        if config_path.exists():
            amil_config = load_config_from_json(str(config_path))
        else:
            # Use default config
            from amil.config import create_default_config
            amil_config, _, _ = create_default_config()

        model = AMILModel(amil_config)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        return {"model": model}

    async def _load_cpg_model(self, config: ModelConfig) -> Dict[str, Any]:
        """Load CPG model."""
        # This would be implemented similarly to AMIL
        # For now, placeholder
        raise NotImplementedError("CPG model loading not implemented in service")

    async def _load_neobert_model(self, config: ModelConfig) -> Dict[str, Any]:
        """Load NeoBERT model."""
        # This would be implemented similarly to AMIL
        # For now, placeholder
        raise NotImplementedError("NeoBERT model loading not implemented in service")

    async def _load_icn_model(self, config: ModelConfig) -> Dict[str, Any]:
        """Load ICN model."""
        # This would be implemented similarly to AMIL
        # For now, placeholder
        raise NotImplementedError("ICN model loading not implemented in service")

    def _estimate_memory_usage(self, model) -> float:
        """Estimate model memory usage in MB."""
        try:
            if hasattr(model, 'parameters'):
                total_params = sum(p.numel() for p in model.parameters())
                # Rough estimate: 4 bytes per parameter (float32)
                return (total_params * 4) / (1024 * 1024)
        except:
            pass
        return 0.0

    async def predict(self, model_name: str, sample: BenchmarkSample) -> BenchmarkResult:
        """Get prediction from a loaded model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        loaded_model = self.models[model_name]

        if loaded_model.status != ModelStatus.READY:
            raise ValueError(f"Model {model_name} not ready: {loaded_model.status}")

        # Increment prediction counter
        loaded_model.prediction_count += 1

        # Route to appropriate prediction method
        config = loaded_model.config

        if config.type == "huggingface":
            return await self._predict_huggingface(loaded_model, sample)
        elif config.type == "amil":
            return await self._predict_amil(loaded_model, sample)
        else:
            raise NotImplementedError(f"Prediction not implemented for {config.type}")

    async def _predict_huggingface(self, loaded_model: LoadedModel, sample: BenchmarkSample) -> BenchmarkResult:
        """Get prediction from HuggingFace model."""
        import time

        start_time = time.time()

        try:
            model = loaded_model.model
            tokenizer = loaded_model.tokenizer

            # Prepare text input
            text_input = sample.raw_content[:8000]  # Truncate for BERT models

            # Tokenize
            inputs = tokenizer(
                text_input,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

                # Convert to prediction
                if logits.dim() > 1 and logits.size(1) > 1:
                    # Binary classification
                    probs = torch.softmax(logits, dim=1)
                    confidence = float(probs.max())
                    prediction = int(logits.argmax(dim=1))
                else:
                    # Single output
                    confidence = float(torch.sigmoid(logits))
                    prediction = int(confidence > 0.5)

            return BenchmarkResult(
                model_name=loaded_model.config.name,
                sample_id=f"{sample.ecosystem}_{sample.package_name}",
                ground_truth=sample.ground_truth_label,
                prediction=prediction,
                confidence=confidence,
                inference_time_seconds=time.time() - start_time,
                success=True
            )

        except Exception as e:
            return BenchmarkResult(
                model_name=loaded_model.config.name,
                sample_id=f"{sample.ecosystem}_{sample.package_name}",
                ground_truth=sample.ground_truth_label,
                prediction=0,
                confidence=0.0,
                inference_time_seconds=time.time() - start_time,
                success=False,
                error_message=str(e)
            )

    async def _predict_amil(self, loaded_model: LoadedModel, sample: BenchmarkSample) -> BenchmarkResult:
        """Get prediction from AMIL model."""
        # This would implement AMIL-specific prediction logic
        # For now, placeholder
        raise NotImplementedError("AMIL prediction not implemented in service")

    async def load_models_from_config(self, configs: List[ModelConfig]) -> Dict[str, bool]:
        """Load multiple models from config list."""
        results = {}

        # Load models concurrently (but be careful about GPU memory)
        for config in configs:
            if config.enabled:
                success = await self.load_model(config)
                results[config.name] = success

        return results

    async def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory."""
        async with self._lock:
            if model_name not in self.models:
                return False

            loaded_model = self.models[model_name]

            # Clean up GPU memory
            if loaded_model.model is not None:
                del loaded_model.model
                if loaded_model.tokenizer is not None:
                    del loaded_model.tokenizer

                # Force garbage collection and clear CUDA cache
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            del self.models[model_name]
            logger.info(f"🗑️ Unloaded model: {model_name}")
            return True

    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all models in the service."""
        status = {
            "device": self.device,
            "total_models": len(self.models),
            "models": {}
        }

        for name, loaded_model in self.models.items():
            status["models"][name] = {
                "status": loaded_model.status.value,
                "type": loaded_model.config.type,
                "load_time_seconds": loaded_model.load_time_seconds,
                "memory_usage_mb": loaded_model.memory_usage_mb,
                "prediction_count": loaded_model.prediction_count,
                "error_message": loaded_model.error_message
            }

        return status

    async def shutdown(self):
        """Shutdown the service and clean up all models."""
        logger.info("🛑 Shutting down ModelService...")

        model_names = list(self.models.keys())
        for name in model_names:
            await self.unload_model(name)

        logger.info("✅ ModelService shutdown complete")


# Global service instance
_model_service: Optional[ModelService] = None


async def get_model_service(device: str = "auto") -> ModelService:
    """Get or create the global model service instance."""
    global _model_service

    if _model_service is None:
        _model_service = ModelService(device)

    return _model_service


async def shutdown_model_service():
    """Shutdown the global model service."""
    global _model_service

    if _model_service is not None:
        await _model_service.shutdown()
        _model_service = None