"""
Prediction Cache System for Model Evaluations
Avoids re-running the same model+prompt+sample combinations.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
import time

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry for a model prediction."""
    model_name: str
    sample_id: str
    prompt_strategy: str
    prediction: int
    confidence: float
    reasoning: str
    malicious_indicators: list
    inference_time_seconds: float
    cost_usd: float
    success: bool
    error_message: Optional[str]
    timestamp: float
    content_hash: str  # Hash of the input content


class PredictionCache:
    """Cache system for model predictions to avoid redundant API calls."""

    def __init__(self, cache_dir: str = "evaluation_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "prediction_cache.json"

        # In-memory cache for fast lookups
        self._cache: Dict[str, CacheEntry] = {}

        # Load existing cache
        self._load_cache()

        # Stats
        self.hits = 0
        self.misses = 0

        logger.info(f"📦 PredictionCache initialized with {len(self._cache)} entries")

    def _create_cache_key(self, model_name: str, sample_id: str, prompt_strategy: str,
                         content_hash: str) -> str:
        """Create a unique cache key for the prediction."""
        key_data = f"{model_name}:{sample_id}:{prompt_strategy}:{content_hash}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _hash_content(self, content: str) -> str:
        """Create a hash of the content to detect changes."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, model_name: str, sample_id: str, prompt_strategy: str,
            content: str) -> Optional[CacheEntry]:
        """Get cached prediction if available."""
        content_hash = self._hash_content(content)
        cache_key = self._create_cache_key(model_name, sample_id, prompt_strategy, content_hash)

        if cache_key in self._cache:
            self.hits += 1
            entry = self._cache[cache_key]
            logger.debug(f"🎯 Cache HIT: {model_name}:{sample_id}:{prompt_strategy}")
            return entry

        self.misses += 1
        logger.debug(f"❌ Cache MISS: {model_name}:{sample_id}:{prompt_strategy}")
        return None

    def put(self, model_name: str, sample_id: str, prompt_strategy: str,
            content: str, prediction: int, confidence: float, reasoning: str,
            malicious_indicators: list, inference_time: float, cost_usd: float,
            success: bool, error_message: Optional[str] = None) -> None:
        """Store a prediction in the cache."""
        content_hash = self._hash_content(content)
        cache_key = self._create_cache_key(model_name, sample_id, prompt_strategy, content_hash)

        entry = CacheEntry(
            model_name=model_name,
            sample_id=sample_id,
            prompt_strategy=prompt_strategy,
            prediction=prediction,
            confidence=confidence,
            reasoning=reasoning,
            malicious_indicators=malicious_indicators,
            inference_time_seconds=inference_time,
            cost_usd=cost_usd,
            success=success,
            error_message=error_message,
            timestamp=time.time(),
            content_hash=content_hash
        )

        self._cache[cache_key] = entry
        logger.debug(f"💾 Cache STORE: {model_name}:{sample_id}:{prompt_strategy}")

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if not self.cache_file.exists():
            return

        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)

            for cache_key, entry_data in cache_data.items():
                # Convert dict back to CacheEntry
                entry = CacheEntry(**entry_data)
                self._cache[cache_key] = entry

            logger.info(f"📂 Loaded {len(self._cache)} cached predictions from {self.cache_file}")

        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    def save_cache(self) -> None:
        """Save cache to disk."""
        try:
            cache_data = {}
            for cache_key, entry in self._cache.items():
                cache_data[cache_key] = asdict(entry)

            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

            logger.info(f"💾 Saved {len(self._cache)} cached predictions to {self.cache_file}")

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def clear_cache(self) -> None:
        """Clear all cached predictions."""
        self._cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("🗑️ Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0

        # Calculate savings
        total_cost_saved = sum(entry.cost_usd for entry in self._cache.values() if entry.success)
        total_time_saved = sum(entry.inference_time_seconds for entry in self._cache.values())

        return {
            "total_entries": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_cost_saved_usd": total_cost_saved,
            "total_time_saved_seconds": total_time_saved
        }

    def cleanup_old_entries(self, max_age_days: int = 30) -> int:
        """Remove cache entries older than specified days."""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

        old_keys = [
            key for key, entry in self._cache.items()
            if entry.timestamp < cutoff_time
        ]

        for key in old_keys:
            del self._cache[key]

        if old_keys:
            logger.info(f"🧹 Cleaned up {len(old_keys)} old cache entries")

        return len(old_keys)


# Global cache instance
_prediction_cache: Optional[PredictionCache] = None


def get_prediction_cache(cache_dir: str = "evaluation_cache") -> PredictionCache:
    """Get or create the global prediction cache."""
    global _prediction_cache

    if _prediction_cache is None:
        _prediction_cache = PredictionCache(cache_dir)

    return _prediction_cache


def save_prediction_cache() -> None:
    """Save the global prediction cache."""
    global _prediction_cache

    if _prediction_cache is not None:
        _prediction_cache.save_cache()


def clear_prediction_cache() -> None:
    """Clear the global prediction cache."""
    global _prediction_cache

    if _prediction_cache is not None:
        _prediction_cache.clear_cache()


if __name__ == "__main__":
    # Test the cache system
    import time

    print("🧪 Testing PredictionCache")

    cache = PredictionCache("test_cache")

    # Test cache miss
    result = cache.get("test_model", "sample_1", "zero_shot", "test content")
    print(f"Cache miss result: {result}")

    # Store in cache
    cache.put(
        model_name="test_model",
        sample_id="sample_1",
        prompt_strategy="zero_shot",
        content="test content",
        prediction=1,
        confidence=0.8,
        reasoning="Test reasoning",
        malicious_indicators=["test"],
        inference_time=1.5,
        cost_usd=0.01,
        success=True
    )

    # Test cache hit
    result = cache.get("test_model", "sample_1", "zero_shot", "test content")
    print(f"Cache hit result: {result}")

    # Test stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")

    # Clean up
    cache.clear_cache()
    print("✅ Cache test completed")