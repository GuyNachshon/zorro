#!/usr/bin/env python3
"""
Multi-Prompt Benchmarking Framework for External Models
Tests all OpenRouter models with multiple prompt strategies and compares performance.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# ICN imports
from icn.evaluation.benchmark_framework import BaseModel, BenchmarkSample, BenchmarkResult, BenchmarkSuite
from icn.evaluation.openrouter_client import OpenRouterClient, BenchmarkRequest, MaliciousPackagePrompts
from icn.evaluation.llm_response_parser import LLMResponseParser
from icn.evaluation.prepare_benchmark_data import BenchmarkDataPreparator
from evaluation.prediction_cache import get_prediction_cache

logger = logging.getLogger(__name__)


@dataclass
class PromptStrategy:
    """Configuration for a prompt testing strategy."""
    name: str
    prompt_type: str  # zero_shot, few_shot, reasoning, file_by_file
    description: str
    granularity: str = "package"  # package, file_by_file
    requires_reasoning: bool = False
    examples_needed: bool = False
    max_tokens: int = 1000
    
    def is_compatible_with_model(self, model_id: str) -> bool:
        """Check if this prompt strategy is compatible with the model."""
        # o1 models work best with reasoning prompts
        if "o1" in model_id.lower() and self.prompt_type != "reasoning":
            return False
        # Non-reasoning models shouldn't use reasoning prompts
        if "o1" not in model_id.lower() and self.requires_reasoning:
            return False
        return True


class MultiPromptOpenRouterModel(BaseModel):
    """OpenRouter model that tests multiple prompt strategies."""
    
    def __init__(self, model_name: str, openrouter_model_id: str, prompt_strategies: List[PromptStrategy]):
        super().__init__(f"{model_name}_MultiPrompt")
        self.original_model_name = model_name
        self.openrouter_model_id = openrouter_model_id
        self.prompt_strategies = [s for s in prompt_strategies if s.is_compatible_with_model(openrouter_model_id)]
        self.response_parser = LLMResponseParser()
        self.examples = []  # For few-shot prompting
        
        logger.info(f"✅ {self.model_name}: {len(self.prompt_strategies)} compatible prompt strategies")
    
    def set_few_shot_examples(self, examples: List[Dict[str, Any]]):
        """Set examples for few-shot prompting."""
        self.examples = examples
    
    async def predict_all_strategies(self, sample: BenchmarkSample, 
                                   openrouter_client: OpenRouterClient) -> List[BenchmarkResult]:
        """Test all prompt strategies for this model."""
        results = []
        
        for strategy in self.prompt_strategies:
            try:
                result = await self._predict_with_strategy(sample, strategy, openrouter_client)
                result.model_name = f"{self.original_model_name}_{strategy.name}"
                result.metadata.update({
                    "prompt_strategy": strategy.name,
                    "prompt_type": strategy.prompt_type,
                    "granularity": strategy.granularity
                })
                results.append(result)
                
                # Small delay between strategies to respect rate limits
                await asyncio.sleep(0.2)
                
            except Exception as e:
                logger.warning(f"Strategy {strategy.name} failed for {self.original_model_name}: {e}")
                # Create failed result
                results.append(BenchmarkResult(
                    model_name=f"{self.original_model_name}_{strategy.name}",
                    sample_id=f"{sample.ecosystem}_{sample.package_name}",
                    ground_truth=sample.ground_truth_label,
                    prediction=0,
                    confidence=0.0,
                    inference_time_seconds=0.0,
                    success=False,
                    error_message=str(e),
                    metadata={"prompt_strategy": strategy.name, "prompt_type": strategy.prompt_type}
                ))
        
        return results
    
    async def _predict_with_strategy(self, sample: BenchmarkSample, strategy: PromptStrategy,
                                   openrouter_client: OpenRouterClient) -> BenchmarkResult:
        """Make prediction using specific prompt strategy."""
        start_time = time.time()

        # Check cache first
        cache = get_prediction_cache()
        sample_id = f"{sample.ecosystem}_{sample.package_name}"
        cached_result = cache.get(self.original_model_name, sample_id, strategy.name, sample.raw_content)

        if cached_result:
            logger.debug(f"🎯 Cache hit for {self.original_model_name}:{sample_id}:{strategy.name}")
            return BenchmarkResult(
                model_name=self.model_name,
                sample_id=sample_id,
                ground_truth=sample.ground_truth_label,
                prediction=cached_result.prediction,
                confidence=cached_result.confidence,
                inference_time_seconds=cached_result.inference_time_seconds,
                cost_usd=cached_result.cost_usd,
                explanation=cached_result.reasoning,
                malicious_indicators=cached_result.malicious_indicators,
                success=cached_result.success,
                error_message=cached_result.error_message,
                metadata={
                    "prompt_strategy": strategy.name,
                    "prompt_type": strategy.prompt_type,
                    "granularity": strategy.granularity,
                    "cache_hit": True
                }
            )

        # Generate appropriate prompt based on strategy
        if strategy.prompt_type == "zero_shot":
            prompt = MaliciousPackagePrompts.zero_shot_prompt(sample.raw_content)
        elif strategy.prompt_type == "few_shot" and self.examples:
            prompt = MaliciousPackagePrompts.few_shot_prompt(sample.raw_content, self.examples)
        elif strategy.prompt_type == "reasoning":
            prompt = MaliciousPackagePrompts.reasoning_prompt(sample.raw_content)
        elif strategy.prompt_type == "file_by_file" and strategy.granularity == "file_by_file":
            return await self._predict_file_by_file_strategy(sample, strategy, openrouter_client)
        else:
            # Fallback to zero-shot
            prompt = MaliciousPackagePrompts.zero_shot_prompt(sample.raw_content)
        
        # Make request
        request = BenchmarkRequest(
            prompt=prompt,
            model_name=self.openrouter_model_id,
            temperature=0.0,
            max_tokens=strategy.max_tokens,
            metadata={
                "sample_id": f"{sample.ecosystem}_{sample.package_name}",
                "prompt_strategy": strategy.name
            }
        )
        
        llm_response = await openrouter_client.generate_response(request)
        
        if not llm_response.success:
            # Store failed request in cache too
            inference_time = time.time() - start_time
            cache.put(
                model_name=self.original_model_name,
                sample_id=sample_id,
                prompt_strategy=strategy.name,
                content=sample.raw_content,
                prediction=0,
                confidence=0.0,
                reasoning="API request failed",
                malicious_indicators=[],
                inference_time=inference_time,
                cost_usd=llm_response.cost_usd,
                success=False,
                error_message=llm_response.error_message or "LLM request failed"
            )

            return BenchmarkResult(
                model_name=self.model_name,
                sample_id=sample_id,
                ground_truth=sample.ground_truth_label,
                prediction=0,
                confidence=0.0,
                inference_time_seconds=inference_time,
                cost_usd=llm_response.cost_usd,
                success=False,
                error_message=llm_response.error_message or "LLM request failed",
                raw_output=llm_response.response_text,
                metadata={
                    "prompt_strategy": strategy.name,
                    "prompt_type": strategy.prompt_type,
                    "granularity": strategy.granularity,
                    "cache_hit": False
                }
            )
        
        # Parse response
        parsed = self.response_parser.parse_response(llm_response.response_text, self.original_model_name)

        # Store in cache
        prediction = 1 if parsed.is_malicious else 0
        inference_time = time.time() - start_time
        cache.put(
            model_name=self.original_model_name,
            sample_id=sample_id,
            prompt_strategy=strategy.name,
            content=sample.raw_content,
            prediction=prediction,
            confidence=parsed.confidence,
            reasoning=parsed.reasoning,
            malicious_indicators=parsed.malicious_indicators,
            inference_time=inference_time,
            cost_usd=llm_response.cost_usd,
            success=True
        )

        return BenchmarkResult(
            model_name=self.model_name,
            sample_id=sample_id,
            ground_truth=sample.ground_truth_label,
            prediction=prediction,
            confidence=parsed.confidence,
            inference_time_seconds=inference_time,
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
                "prompt_strategy": strategy.name,
                "prompt_type": strategy.prompt_type,
                "granularity": strategy.granularity,
                "cache_hit": False
            }
        )
    
    async def _predict_file_by_file_strategy(self, sample: BenchmarkSample, strategy: PromptStrategy,
                                           openrouter_client: OpenRouterClient) -> BenchmarkResult:
        """Handle file-by-file analysis strategy with early stopping."""
        start_time = time.time()

        if not sample.individual_files:
            logger.warning(f"No individual files for {sample.package_name}, skipping file-by-file")
            return BenchmarkResult(
                model_name=self.model_name,
                sample_id=f"{sample.ecosystem}_{sample.package_name}",
                ground_truth=sample.ground_truth_label,
                prediction=0,
                confidence=0.0,
                inference_time_seconds=time.time() - start_time,
                success=False,
                error_message="No individual files available for file-by-file analysis"
            )

        # Analyze up to 6 files to control costs (with early stopping)
        files_to_analyze = list(sample.individual_files.items())[:6]
        file_analyses = []
        total_cost = 0.0
        found_malicious = False

        for file_path, file_content in files_to_analyze:
            if not file_content.strip():
                continue

            prompt = MaliciousPackagePrompts.file_by_file_prompt(file_path, file_content)
            request = BenchmarkRequest(
                prompt=prompt,
                model_name=self.openrouter_model_id,
                temperature=0.0,
                max_tokens=600,
                metadata={"file_path": file_path, "sample_id": f"{sample.ecosystem}_{sample.package_name}"}
            )

            llm_response = await openrouter_client.generate_response(request)

            if llm_response.success:
                parsed = self.response_parser.parse_response(llm_response.response_text, self.original_model_name)
                file_analyses.append({
                    "file_path": file_path,
                    "is_malicious": parsed.is_malicious,
                    "confidence": parsed.confidence,
                    "reasoning": parsed.reasoning,
                    "malicious_indicators": parsed.malicious_indicators
                })
                total_cost += llm_response.cost_usd

                # Early stopping: if we found malicious content with high confidence, stop
                if parsed.is_malicious and parsed.confidence >= 0.8:
                    found_malicious = True
                    logger.info(f"🛑 Early stopping: Found malicious content in {file_path} with confidence {parsed.confidence:.2f}")
                    break

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
        
        # Aggregate results
        malicious_files = [f for f in file_analyses if f.get("is_malicious", False)]
        is_package_malicious = len(malicious_files) > 0
        total_files_available = len(files_to_analyze)

        if is_package_malicious:
            confidence = max([f.get("confidence", 0) for f in malicious_files])
            if found_malicious:
                explanation = f"File-by-file (early stop): {len(malicious_files)}/{len(file_analyses)} files flagged, stopped at first malicious"
            else:
                explanation = f"File-by-file: {len(malicious_files)}/{len(file_analyses)} files flagged"
        else:
            benign_confidences = [f.get("confidence", 0.5) for f in file_analyses if not f.get("is_malicious", False)]
            confidence = np.mean(benign_confidences) if benign_confidences else 0.5
            explanation = f"File-by-file: 0/{len(file_analyses)} files analyzed, all benign"

        return BenchmarkResult(
            model_name=self.model_name,
            sample_id=f"{sample.ecosystem}_{sample.package_name}",
            ground_truth=sample.ground_truth_label,
            prediction=1 if is_package_malicious else 0,
            confidence=min(1.0, max(0.0, confidence)),
            inference_time_seconds=time.time() - start_time,
            cost_usd=total_cost,
            explanation=explanation,
            success=True,
            metadata={
                "files_analyzed": len(file_analyses),
                "total_files_available": total_files_available,
                "malicious_files": len(malicious_files),
                "early_stopped": found_malicious,
                "prompt_strategy": strategy.name,
                "prompt_type": strategy.prompt_type,
                "granularity": "file_by_file"
            }
        )
    
    async def predict(self, sample: BenchmarkSample) -> BenchmarkResult:
        """Default predict method - not used in multi-prompt mode."""
        raise NotImplementedError("Use predict_all_strategies() for multi-prompt testing")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": "MultiPrompt_OpenRouter_LLM",
            "openrouter_model_id": self.openrouter_model_id,
            "prompt_strategies": [s.name for s in self.prompt_strategies],
            "supports_explanations": True,
            "inference_method": "api"
        }


class PromptEffectivenessAnalyzer:
    """Analyzes the effectiveness of different prompt strategies."""
    
    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
        self.results_df = self._create_dataframe()
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        data = []
        for result in self.results:
            if result.success:
                data.append({
                    'model_name': result.model_name.split('_')[0] if '_' in result.model_name else result.model_name,
                    'full_model_name': result.model_name,
                    'prompt_strategy': result.metadata.get('prompt_strategy', 'unknown'),
                    'prompt_type': result.metadata.get('prompt_type', 'unknown'),
                    'granularity': result.metadata.get('granularity', 'package'),
                    'sample_id': result.sample_id,
                    'ground_truth': result.ground_truth,
                    'prediction': result.prediction,
                    'confidence': result.confidence,
                    'inference_time': result.inference_time_seconds,
                    'cost_usd': result.cost_usd or 0.0,
                    'success': result.success
                })
        return pd.DataFrame(data)
    
    def analyze_prompt_effectiveness(self) -> Dict[str, Any]:
        """Analyze which prompt strategies work best for each model."""
        if self.results_df.empty:
            return {"error": "No successful results to analyze"}
        
        # Group by model and prompt strategy
        grouped = self.results_df.groupby(['model_name', 'prompt_strategy']).agg({
            'ground_truth': 'count',  # Number of samples
            'prediction': lambda x: ((x == self.results_df.loc[x.index, 'ground_truth']).sum() / len(x)),  # Accuracy
            'confidence': 'mean',
            'inference_time': 'mean',
            'cost_usd': 'sum'
        }).round(4)
        
        grouped.columns = ['samples', 'accuracy', 'avg_confidence', 'avg_time', 'total_cost']
        
        # Find best strategy per model
        best_strategies = {}
        for model in self.results_df['model_name'].unique():
            model_data = grouped.loc[model] if model in grouped.index.get_level_values(0) else pd.DataFrame()
            if not model_data.empty:
                # Rank by accuracy, then by confidence
                model_data['rank_score'] = model_data['accuracy'] + 0.1 * model_data['avg_confidence']
                best_strategy = model_data['rank_score'].idxmax()
                best_strategies[model] = {
                    'strategy': best_strategy,
                    'accuracy': model_data.loc[best_strategy, 'accuracy'],
                    'confidence': model_data.loc[best_strategy, 'avg_confidence'],
                    'cost': model_data.loc[best_strategy, 'total_cost']
                }
        
        # Convert MultiIndex to string keys for JSON serialization
        strategy_matrix = {}
        for (model, strategy), row in grouped.iterrows():
            key = f"{model}_{strategy}"
            strategy_matrix[key] = {
                'model': model,
                'strategy': strategy,
                'samples': row['samples'],
                'accuracy': row['accuracy'],
                'avg_confidence': row['avg_confidence'],
                'avg_time': row['avg_time'],
                'total_cost': row['total_cost']
            }

        return {
            'strategy_performance_matrix': strategy_matrix,
            'best_strategies_per_model': best_strategies,
            'overall_strategy_ranking': self._rank_strategies_overall()
        }
    
    def _rank_strategies_overall(self) -> List[Dict[str, Any]]:
        """Rank prompt strategies by overall effectiveness."""
        strategy_stats = self.results_df.groupby('prompt_strategy').agg({
            'ground_truth': 'count',
            'prediction': lambda x: ((x == self.results_df.loc[x.index, 'ground_truth']).sum() / len(x)),
            'confidence': 'mean',
            'inference_time': 'mean',
            'cost_usd': 'mean'
        }).round(4)
        
        strategy_stats.columns = ['samples', 'accuracy', 'avg_confidence', 'avg_time', 'avg_cost']
        strategy_stats['efficiency_score'] = strategy_stats['accuracy'] / (strategy_stats['avg_cost'] + 0.001)
        
        # Sort by accuracy, then efficiency
        strategy_stats = strategy_stats.sort_values(['accuracy', 'efficiency_score'], ascending=[False, False])
        
        return [
            {
                'strategy': strategy,
                'accuracy': row['accuracy'],
                'confidence': row['avg_confidence'],
                'avg_time': row['avg_time'],
                'avg_cost': row['avg_cost'],
                'efficiency_score': row['efficiency_score'],
                'samples': int(row['samples'])
            }
            for strategy, row in strategy_stats.iterrows()
        ]
    
    def generate_comparison_report(self) -> str:
        """Generate detailed comparison report."""
        analysis = self.analyze_prompt_effectiveness()
        
        if "error" in analysis:
            return f"## Error\n{analysis['error']}"
        
        report = f"""# Multi-Prompt Benchmarking Analysis Report

Generated: {datetime.now().isoformat()}

## Overall Strategy Ranking

| Strategy | Accuracy | Avg Confidence | Avg Cost ($) | Efficiency Score | Samples |
|----------|----------|----------------|--------------|------------------|---------|
"""
        
        for strategy in analysis['overall_strategy_ranking']:
            report += f"| {strategy['strategy']} | {strategy['accuracy']:.3f} | {strategy['confidence']:.3f} | ${strategy['avg_cost']:.4f} | {strategy['efficiency_score']:.2f} | {strategy['samples']} |\n"
        
        report += "\n## Best Strategy Per Model\n\n"
        
        for model, info in analysis['best_strategies_per_model'].items():
            report += f"**{model}**: {info['strategy']} (Accuracy: {info['accuracy']:.3f}, Cost: ${info['cost']:.4f})\n"
        
        report += "\n## Strategy Performance Matrix\n\n"
        
        # Create performance matrix table
        if analysis['strategy_performance_matrix']:
            models = list(set([k[0] for k in analysis['strategy_performance_matrix'].keys()]))
            strategies = list(set([k[1] for k in analysis['strategy_performance_matrix'].keys()]))
            
            report += "| Model | " + " | ".join(strategies) + " |\n"
            report += "|-------|" + "---|" * len(strategies) + "\n"
            
            for model in models:
                row = [model]
                for strategy in strategies:
                    key = (model, strategy)
                    if key in analysis['strategy_performance_matrix']:
                        accuracy = analysis['strategy_performance_matrix'][key]['accuracy']
                        row.append(f"{accuracy:.3f}")
                    else:
                        row.append("N/A")
                report += "| " + " | ".join(row) + " |\n"
        
        return report


class MultiPromptBenchmarkSuite(BenchmarkSuite):
    """Extended benchmark suite for multi-prompt testing."""
    
    def __init__(self, output_dir: str = "multi_prompt_benchmark_results"):
        super().__init__(output_dir)
        self.prompt_strategies = self._initialize_prompt_strategies()
        self.multi_prompt_results: List[BenchmarkResult] = []
    
    def _initialize_prompt_strategies(self) -> List[PromptStrategy]:
        """Initialize available prompt strategies."""
        return [
            PromptStrategy(
                name="zero_shot",
                prompt_type="zero_shot",
                description="Direct analysis without examples",
                granularity="package",
                max_tokens=1000
            ),
            PromptStrategy(
                name="few_shot",
                prompt_type="few_shot",
                description="Analysis with examples",
                granularity="package",
                examples_needed=True,
                max_tokens=1500
            ),
            PromptStrategy(
                name="reasoning",
                prompt_type="reasoning",
                description="Step-by-step reasoning analysis",
                granularity="package",
                requires_reasoning=True,
                max_tokens=2000
            ),
            PromptStrategy(
                name="file_by_file",
                prompt_type="file_by_file",
                description="Individual file analysis with aggregation",
                granularity="file_by_file",
                max_tokens=800
            )
        ]
    
    def register_openrouter_models(self, openrouter_client: OpenRouterClient, 
                                 model_ids: Optional[List[str]] = None) -> int:
        """Register OpenRouter models for multi-prompt testing."""
        if model_ids is None:
            # Use all available models
            model_ids = list(openrouter_client.models.keys())
        
        registered = 0
        for model_id in model_ids:
            # Auto-add missing models with default config
            if model_id not in openrouter_client.models:
                logger.info(f"➕ Adding missing model config for: {model_id}")
                openrouter_client.add_model_config(model_id)

            model_config = openrouter_client.models[model_id]
            model_name = model_config.name.replace('/', '_')

            # Create multi-prompt model
            multi_prompt_model = MultiPromptOpenRouterModel(
                model_name=model_name,
                openrouter_model_id=model_id,
                prompt_strategies=self.prompt_strategies
            )

            # Store as special multi-prompt model (not in regular models dict)
            if not hasattr(self, 'multi_prompt_models'):
                self.multi_prompt_models = {}
            self.multi_prompt_models[model_name] = multi_prompt_model
            registered += 1

            logger.info(f"📝 Registered multi-prompt model: {model_name}")
        
        return registered
    
    async def run_multi_prompt_benchmark(self, openrouter_client: OpenRouterClient,
                                       max_concurrent: int = 2) -> pd.DataFrame:
        """Run multi-prompt benchmark across all registered models."""
        if not hasattr(self, 'multi_prompt_models') or not self.multi_prompt_models:
            raise ValueError("No multi-prompt models registered. Call register_openrouter_models() first.")
        
        logger.info(f"🚀 Starting multi-prompt benchmark: {len(self.multi_prompt_models)} models × {len(self.samples)} samples")
        
        # Set up few-shot examples if any strategies need them
        self._prepare_few_shot_examples()
        
        all_results = []
        
        for model_name, model in self.multi_prompt_models.items():
            logger.info(f"🤖 Running multi-prompt benchmark for {model_name}...")
            
            # Create tasks for this model (each sample tests all strategies)
            tasks = [model.predict_all_strategies(sample, openrouter_client) for sample in self.samples]
            
            # Run with concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def limited_task(task):
                async with semaphore:
                    return await task
            
            limited_tasks = [limited_task(task) for task in tasks]
            model_result_lists = await asyncio.gather(*limited_tasks)
            
            # Flatten results (each task returns a list of results)
            model_results = [result for result_list in model_result_lists for result in result_list]
            
            # Track success rate
            successful = sum(1 for r in model_results if r.success)
            logger.info(f"   ✅ {successful}/{len(model_results)} predictions successful")
            
            all_results.extend(model_results)
        
        self.multi_prompt_results = all_results

        # Save cache after benchmark completion
        cache = get_prediction_cache()
        cache.save_cache()

        # Log cache statistics
        cache_stats = cache.get_stats()
        logger.info(f"📊 Cache Statistics:")
        logger.info(f"   Total entries: {cache_stats['total_entries']}")
        logger.info(f"   Hit rate: {cache_stats['hit_rate']:.1%} ({cache_stats['hits']}/{cache_stats['hits'] + cache_stats['misses']})")
        logger.info(f"   Cost saved: ${cache_stats['total_cost_saved_usd']:.4f}")
        logger.info(f"   Time saved: {cache_stats['total_time_saved_seconds']:.1f}s")

        # Convert to DataFrame for analysis
        return self._multi_prompt_results_to_dataframe()
    
    def _prepare_few_shot_examples(self):
        """Prepare few-shot examples from existing samples."""
        # Use a small subset of samples as examples
        example_samples = self.samples[:3] if len(self.samples) >= 3 else self.samples
        
        examples = []
        for sample in example_samples:
            examples.append({
                "content": sample.raw_content[:1000],  # Truncate for examples
                "is_malicious": bool(sample.ground_truth_label),
                "reasoning": f"This package is {'malicious' if sample.ground_truth_label else 'benign'} based on static analysis."
            })
        
        # Set examples for all models
        if hasattr(self, 'multi_prompt_models'):
            for model in self.multi_prompt_models.values():
                model.set_few_shot_examples(examples)
    
    def _multi_prompt_results_to_dataframe(self) -> pd.DataFrame:
        """Convert multi-prompt results to DataFrame."""
        data = []
        for result in self.multi_prompt_results:
            data.append({
                'model_name': result.model_name,
                'sample_id': result.sample_id,
                'ground_truth': result.ground_truth,
                'prediction': result.prediction,
                'confidence': result.confidence,
                'inference_time': result.inference_time_seconds,
                'cost_usd': result.cost_usd,
                'success': result.success,
                'error_message': result.error_message,
                'prompt_strategy': result.metadata.get('prompt_strategy', 'unknown'),
                'prompt_type': result.metadata.get('prompt_type', 'unknown'),
                'granularity': result.metadata.get('granularity', 'package')
            })
        
        return pd.DataFrame(data)
    
    def analyze_results(self) -> PromptEffectivenessAnalyzer:
        """Create analyzer for multi-prompt results."""
        return PromptEffectivenessAnalyzer(self.multi_prompt_results)
    
    def save_multi_prompt_results(self, filename: str = "multi_prompt_results.json"):
        """Save detailed multi-prompt results."""
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "models": {name: model.get_model_info() for name, model in getattr(self, 'multi_prompt_models', {}).items()},
            "prompt_strategies": [
                {
                    "name": s.name,
                    "prompt_type": s.prompt_type,
                    "description": s.description,
                    "granularity": s.granularity,
                    "requires_reasoning": s.requires_reasoning,
                    "max_tokens": s.max_tokens
                } for s in self.prompt_strategies
            ],
            "samples_count": len(self.samples),
            "total_predictions": len(self.multi_prompt_results),
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
                    "prompt_strategy": r.metadata.get('prompt_strategy'),
                    "prompt_type": r.metadata.get('prompt_type'),
                    "granularity": r.metadata.get('granularity')
                }
                for r in self.multi_prompt_results
            ]
        }
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"💾 Multi-prompt results saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("🧪 Multi-Prompt Benchmarking Framework initialized")
    print("Ready for comprehensive model and prompt strategy comparison!")