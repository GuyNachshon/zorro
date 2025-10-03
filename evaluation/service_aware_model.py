"""
Service-aware model wrappers that use the ModelService instead of loading models directly.
"""

import asyncio
import logging
from typing import Optional
import time

from icn.evaluation.benchmark_framework import BaseModel, BenchmarkSample, BenchmarkResult
from .model_service import get_model_service, ModelService
from .config import ModelConfig

logger = logging.getLogger(__name__)


class ServiceAwareHuggingFaceModel(BaseModel):
    """HuggingFace model that uses the ModelService."""

    def __init__(self, config: ModelConfig):
        super().__init__(config.name)
        self.config = config
        self._service: Optional[ModelService] = None

    async def _ensure_service(self):
        """Ensure model service is available and model is loaded."""
        if self._service is None:
            self._service = await get_model_service()

        # Make sure our model is loaded in the service
        if self.config.name not in self._service.models:
            success = await self._service.load_model(self.config)
            if not success:
                raise RuntimeError(f"Failed to load model {self.config.name} in service")

    async def predict(self, sample: BenchmarkSample) -> BenchmarkResult:
        """Get prediction using the model service."""
        await self._ensure_service()
        return await self._service.predict(self.config.name, sample)

    def get_model_info(self) -> dict:
        """Get model information (required by BaseModel)."""
        return self.get_metadata()

    def get_metadata(self) -> dict:
        """Get model metadata."""
        return {
            "model_name": self.model_name,
            "type": self.config.type,
            "use_peft": getattr(self.config, 'use_peft', False),
            "base_model_id": getattr(self.config, 'hf_base_model_id', None),
            "adapter_id": getattr(self.config, 'hf_adapter_id', None),
            "model_id": getattr(self.config, 'hf_model_id', None),
            "inference_method": "model_service"
        }


class ServiceAwareAMILModel(BaseModel):
    """AMIL model that uses the ModelService."""

    def __init__(self, config: ModelConfig):
        super().__init__(config.name)
        self.config = config
        self._service: Optional[ModelService] = None

    async def _ensure_service(self):
        """Ensure model service is available and model is loaded."""
        if self._service is None:
            self._service = await get_model_service()

        # Make sure our model is loaded in the service
        if self.config.name not in self._service.models:
            success = await self._service.load_model(self.config)
            if not success:
                raise RuntimeError(f"Failed to load model {self.config.name} in service")

    async def predict(self, sample: BenchmarkSample) -> BenchmarkResult:
        """Get prediction using the model service."""
        await self._ensure_service()
        return await self._service.predict(self.config.name, sample)

    def get_model_info(self) -> dict:
        """Get model information (required by BaseModel)."""
        return self.get_metadata()

    def get_metadata(self) -> dict:
        """Get model metadata."""
        return {
            "model_name": self.model_name,
            "type": self.config.type,
            "model_path": self.config.model_path,
            "inference_method": "model_service"
        }


def create_service_aware_model(config: ModelConfig) -> BaseModel:
    """Factory function to create service-aware models."""
    if config.type == "huggingface":
        return ServiceAwareHuggingFaceModel(config)
    elif config.type == "amil":
        return ServiceAwareAMILModel(config)
    # Add other model types as needed
    else:
        raise ValueError(f"Service-aware model not implemented for type: {config.type}")